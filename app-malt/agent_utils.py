import httpx
from loguru import logger
from uuid import uuid4
from typing import Any
from enum import Enum
from dataclasses import dataclass, field

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message, new_data_artifact, new_task
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    MessageSendParams,
    SendMessageRequest,
    SendMessageResponse,
    SendMessageSuccessResponse,
    Task,
    Message,
    TextPart,
    Role,
    SendStreamingMessageRequest,
)
from a2a.utils.constants import (
    AGENT_CARD_WELL_KNOWN_PATH,
    EXTENDED_AGENT_CARD_PATH,
)


class PromptType(Enum):
    ZEROSHOT_BASE = "zeroshot_base"
    FEWSHOT_BASE = "fewshot_base"
    ZEROSHOT_COT = "zeroshot_cot"
    FEWSHOT_COT = "fewshot_cot"


@dataclass
class AgentServerConfig:
    name: str
    base_url: str
    prompt_type: PromptType = PromptType.FEWSHOT_COT
    http_kwargs: dict[str, Any] = field(default_factory=dict)


class AgentServer:
    def __init__(self, config: AgentServerConfig):
        self.config = config

    async def start(self, http_client: httpx.AsyncClient | None = None):
        logger.info("Starting Agent Server...")
        # Initialize the A2AClient with the provided configuration
        self.http_client = httpx.AsyncClient() if not http_client else http_client

        # Fetch and set the agent card.
        self.agent_card = await get_agent_card(self.http_client, self.config.base_url)
        self.a2a_client = A2AClient(
            httpx_client=self.http_client,
            agent_card=self.agent_card
        )
        # Start the A2AClient
        logger.info("Agent Server started successfully.")

    async def handle_query(self, query_text: str) -> str | None:
        try:
            prompt = process_query(query_text, self.config.prompt_type)
            logger.debug(f"Processing query with prompt: {prompt}")
            response = await call_a2a_agent(self.a2a_client, prompt, http_kwargs=self.config.http_kwargs)
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            response = None
        return response

    async def stop(self):
        logger.info("Stopping Agent Server...")
        # Stop the A2AClient
        await self.http_client.aclose()
        logger.info("Agent Server stopped successfully.")


BASE_PROMPT = """
You need to behave like a network engineer who processes graph data to answer user queries about capacity planning.

Your task is to generate the Python code needed to process the network graph to answer the user question or request. The code should take the form of a function named process_graph that accepts a single input argument graph_data and returns a single object return_object.

Graph Structure:
- The input graph_data is a networkx graph object with nodes and edges
- The graph is directed and each node has a 'name' attribute to represent itself
- Each node has a 'type' attribute in the format of EK_TYPE. 'type' must be a list, which can include ['EK_SUPERBLOCK', 'EK_CHASSIS', 'EK_RACK', 'EK_AGG_BLOCK', 'EK_JUPITER', 'EK_PORT', 'EK_SPINEBLOCK', 'EK_PACKET_SWITCH', 'EK_CONTROL_POINT', 'EK_CONTROL_DOMAIN']
- Each node can have other attributes depending on its type
- Each directed edge also has a 'type' attribute, including RK_CONTAINS, RK_CONTROL

Important Guidelines:
- Check relationships based on edge, check name based on node attribute
- Nodes follow this hierarchy: CHASSIS contains PACKET_SWITCH, JUPITER contains SUPERBLOCK, SUPERBLOCK contains AGG_BLOCK, AGG_BLOCK contains PACKET_SWITCH, PACKET_SWITCH contains PORT
- Each PORT node has an attribute 'physical_capacity_bps'. For example, a PORT node name is ju1.a1.m1.s2c1.p3
- When calculating capacity of a node, sum the physical_capacity_bps on the PORT of each hierarchy contained in this node
- When updating a graph, always create a graph copy, do not modify the input graph
- To find a node based on type, check the name and type list. For example, [node[0] == 'ju1.a1.m1.s2c1' and 'EK_PACKET_SWITCH' in node[1]['type']]
- node.startswith will not work for the node name. You have to check the node name with the node['name']


Output Format:
- Do not use multi-layer functions. The output format should only return one object
- The return_object must be a JSON object with three keys: 'type', 'data', and 'updated_graph'
- The 'type' key should indicate the output format depending on the user query or request. It should be one of 'text', 'list', 'table', or 'graph'
- The 'data' key should contain the data needed to render the output:
    * If output type is 'text': 'data' should contain a string
    * If output type is 'list': 'data' should contain a list of items
    * If output type is 'table': 'data' should contain a list of lists where each list represents a row in the table
    * If output type is 'graph': 'data' should contain a graph JSON
- The 'updated_graph' key should always contain the updated graph as "graph_json = nx.readwrite.json_graph.node_link_data(graph_copy)"
    
Response Format:
- Your reply should always start with string "\\nAnswer:\\n"
- You should generate a function called "def process_graph"
- All of your output should only contain the defined function without example usages, no additional text, and displayed in a Python code block
- Do not include any package imports in your answer
"""

COT_PROMPT = """Please think step by step and provide your output."""

QA_TEMPLATE = """
Question: {input}

Answer:
```python
${{Code that will answer the user question or request}}
```
"""

FEWSHOT_PROMPT = f"""Here are a few examples of questions and their corresponding answers:"""

PROMPT_SUFFIX = f"""Begin! Remember to ensure that you generate valid Python code in the following format: {QA_TEMPLATE}"""

EXAMPLE_LIST = [
            {
                "question": "Update the physical capacity value of ju1.a3.m2.s2c4.p10 to 72. Return a graph.",
                "answer": r'''def process_graph(graph_data):    
                                graph_copy = copy.deepcopy(graph_data)    
                                for node in graph_copy.nodes(data=True):        
                                    if node[1]['name'] == 'ju1.a3.m2.s2c4.p10' and 'EK_PORT' in node[1]['type']:            
                                        node[1]['physical_capacity_bps'] = 72           
                                    break    
                                graph_json = nx.readwrite.json_graph.node_link_data(graph_copy)    
                                # in the return_object, it should be a json object with three keys, 'type', 'data' and 'updated_graph'.
                                return return_object''',
            },
            {
                "question": "Add new node with name new_EK_PORT_82 type EK_PORT, to ju1.a2.m4.s3c6. Return a graph.",
                "answer": r'''def process_graph(graph_data):
                                graph_copy = copy.deepcopy(graph_data)
                                graph_copy.add_node('new_EK_PORT_82', type=['EK_PORT'], physical_capacity_bps=1000)
                                graph_copy.add_edge('ju1.a2.m4.s3c6', 'new_EK_PORT_82', type=['RK_CONTAINS'])
                                graph_json = nx.readwrite.json_graph.node_link_data(graph_copy)  
                                # in the return_object, it should be a json object with three keys, 'type', 'data' and 'updated_graph'. 
                                return return_object''',
            },
            {
                "question": "Count the EK_PACKET_SWITCH in the ju1.a2.dom. Return only the count number.",
                "answer": r'''def process_graph(graph_data):
                                graph_copy = graph_data.copy()
                                count = 0
                                for node in graph_copy.nodes(data=True):
                                    if 'EK_PACKET_SWITCH' in node[1]['type'] and node[0].startswith('ju1.a2.'):
                                        count += 1
                                # the return_object should be a json object with three keys, 'type', 'data' and 'updated_graph'. 
                                return return_object''',
            },
            {
                "question": "Remove ju1.a1.m4.s3c6.p1 from the graph. Return a graph.",
                "answer": r'''def process_graph(graph_data):
                                graph_copy = graph_data.copy()
                                node_to_remove = None
                                for node in graph_copy.nodes(data=True):
                                    if node[0] == 'ju1.a1.m4.s3c6.p1':
                                        node_to_remove = node[0]
                                        break
                                if node_to_remove:
                                    graph_copy.remove_node(node_to_remove)
                                graph_json = nx.readwrite.json_graph.node_link_data(graph_copy)
                                # in the return_object, it should be a json object with three keys, 'type', 'data' and 'updated_graph'. 
                                return return_object''',
            },
        ]


async def get_agent_card(http_client: httpx.AsyncClient, base_url: str):
    resolver = A2ACardResolver(
        httpx_client=http_client,
        base_url=base_url,
        # agent_card_path uses default, extended_agent_card_path also uses default
    )
    try:
        logger.info(f'Attempting to fetch public agent card from: {base_url}{AGENT_CARD_WELL_KNOWN_PATH}')
        public_card = await resolver.get_agent_card()  # Fetches from default public path
        logger.info('Successfully fetched public agent card:')
        logger.debug(public_card.model_dump_json(indent=2, exclude_none=True))
        final_agent_card_to_use = public_card
    except Exception as e:
        logger.error(f'Failed to fetch public agent card: {e}') 
        final_agent_card_to_use = None
    return final_agent_card_to_use


async def call_a2a_agent(a2a_client: A2AClient, query_text: str, http_kwargs: dict = {}) -> str | None:
    send_message_payload: dict[str, Any] = {
        'message': {
            'role': 'user',
            'parts': [
                {'kind': 'text', 'text': query_text}
            ],
            'messageId': uuid4().hex,
        },
    }
    request = SendMessageRequest(id=str(uuid4()), params=MessageSendParams(**send_message_payload))
    response = await a2a_client.send_message(request, http_kwargs=http_kwargs)
    response_json = response.model_dump_json(indent=2, exclude_none=True)
    logger.debug(f'Received response from agent: {response_json}')
    llm_answer = _extract_text_from_response(response)
    return llm_answer


def process_query(query_text: str, prompt_type: PromptType) -> str:
    subsituted_query_text = PROMPT_SUFFIX.format(input=query_text)   
    if prompt_type == PromptType.ZEROSHOT_COT:
        prompt = '\n'.join([BASE_PROMPT, COT_PROMPT, subsituted_query_text])
    elif prompt_type == PromptType.FEWSHOT_COT:
        examples = [QA_TEMPLATE.format(input=ex['question'], answer=ex['answer']) for ex in EXAMPLE_LIST]
        prompt = '\n'.join([BASE_PROMPT, FEWSHOT_PROMPT, *examples, subsituted_query_text])
    elif prompt_type == PromptType.FEWSHOT_BASE:
        examples = [QA_TEMPLATE.format(input=ex['question'], answer=ex['answer']) for ex in EXAMPLE_LIST]
        prompt = '\n'.join([BASE_PROMPT, *examples, subsituted_query_text])
    else:
        prompt = BASE_PROMPT + PROMPT_SUFFIX
    return prompt


def _extract_text_from_response(response: SendMessageResponse) -> str | None:
    root = response.root
    if isinstance(root, SendMessageSuccessResponse):
        result = root.result
        logger.debug(f'Extracting text from result: {result}')
        if isinstance(result, Message):
            return _extract_text_from_message(result)
        elif isinstance(result, Task):
            return _extract_text_from_task(result)
    return None


def _extract_text_from_message(message: Message) -> str | None:
    if isinstance(message, dict):
        message = Message.model_validate(message)
    parts = message.parts
    text_parts = [part.root.text for part in parts if isinstance(part.root, TextPart)]
    text = ''.join(text_parts).strip() if text_parts else None
    return text


def _extract_text_from_task(task: Task | dict) -> str | None:
    """Extract a text reply from a Task model or mapping by inspecting its history and artifacts."""
    if isinstance(task, dict):
        task = Task.model_validate(task)

    # Try artifacts first (prefer last artifact with text)
    artifacts = task.artifacts if task.artifacts else []
    for art in reversed(artifacts):
        # Some artifacts may include parts or textual content. Return first one found.
        text_parts = [part.root.text for part in art.parts if isinstance(part.root, TextPart)]
        return ''.join(text_parts).strip()
            
    # Try history first (prefer latest agent message with text).
    history = task.history if task.history else []
    for msg in reversed(history):
        if msg.role == Role.agent:  
            text = _extract_text_from_message(msg)
            if text:
                return text
    return None
