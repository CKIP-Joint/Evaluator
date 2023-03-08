from src.helm.benchmark.scenarios.drcd_scenario import DRCDScenario
from src.helm.benchmark.scenarios.fgc_scenario import FGCScenario
from src.helm.benchmark.scenarios.tcwsc_scenario import TCWSCScenario
from src.helm.benchmark.scenarios.tcic_scenario import TCICScenario
from src.helm.benchmark.scenarios.sltp_scenario import SLTPScenario
from src.helm.benchmark.scenarios.lambada_scenario import LAMBADAScenario


if __name__ == "__main__":
    s = LAMBADAScenario()
    s.get_instances()
    s = SLTPScenario()
    s.get_instances()
    s = TCICScenario()
    s.get_instances()
    s = TCWSCScenario()
    s.get_instances()
    s = DRCDScenario()
    s.get_instances()
    s = FGCScenario()
    s.get_instances()