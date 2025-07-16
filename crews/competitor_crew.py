import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool
#from tools.custom_tools import SerperDevTool

from tools.custom_tools import DistanceCalculatorTool, CompetitorVerifierTool

load_dotenv()

# Configurar el modelo LLM
llm = LLM(
    model='gemini/gemini-2.0-flash-lite',
    api_key=os.environ["GEMINI_API_KEY"],
    temperature=0,
    verbose=True,
)

# Herramientas
serper = SerperDevTool()
distance_tool = DistanceCalculatorTool()
verifier_tool = CompetitorVerifierTool()

# Agentes
dealership_info_agent = Agent(
    role="Dealership Info Gatherer",
    goal="Collect key information about the target car dealership {dealership} with a zipcode {zipcode}.",
    backstory="Expert researcher skilled at finding dealership details.",
    llm=llm,
    verbose=True
)

competitor_researcher = Agent(
    role="Competitor Researcher",
    goal="Identify competitor car dealerships for {dealership}.",
    backstory="Market research specialist focused on car dealerships.",
    llm=llm,
    verbose=True
)

data_organizer = Agent(
    role="Data Organizer",
    goal="Structure competitor data into a table with dealership info.",
    backstory="Data analyst experienced in formatting and cleaning information.",
    llm=llm,
    verbose=True
)

results_supervisor = Agent(
    role="Results Supervisor",
    goal="Validate dealership and competitor data for accuracy and quality.",
    backstory="Quality assurance expert reviewing all information.",
    llm=llm,
    verbose=True
)

# Tareas
dealership_info_task = Task(
    description="Find the OEM, address, and coordinates of the dealership {dealership} with zipcode {zipcode}.",
    expected_output="Dictionary with OEM, address, and coordinates.",
    agent=dealership_info_agent,
    tools=[serper]
)

competitor_research_task = Task(
    description="Search competitors selling the same OEM within 100 miles of {dealership}.",
    expected_output="List of competitors with name, website, address, city/state and distance.",
    agent=competitor_researcher,
    tools=[serper, distance_tool]
)

data_organization_task = Task(
    description="Organize competitor data into a structured table.",
    expected_output="Markdown table: Name | Website | Distance | City | State.",
    agent=data_organizer
)

supervision_task = Task(
    description="Verify all competitor data, check OEM, website, and existence.",
    expected_output="Final summary with dealership info and validated competitor table.",
    agent=results_supervisor,
    tools=[verifier_tool]
)

# Crew
competitor_crew = Crew(
    agents=[dealership_info_agent, competitor_researcher, data_organizer, results_supervisor],
    tasks=[dealership_info_task, competitor_research_task, data_organization_task, supervision_task],
    process=Process.sequential,
   #poner true
    verbose=True
)

# Función para ejecutar desde main.py
def run_competitor_crew(**kwargs):
    return competitor_crew.kickoff(inputs=kwargs)
