from .attribution import handle_attribution_workflow
from .design import handle_design_workflow
from .info import handle_info_workflow
from .predict import handle_predict_workflow
from .vep import handle_vep_workflow
from .scan import handle_scan_workflow
from .zoo import handle_zoo_workflow

WORKFLOWS = {
    "info": handle_info_workflow,
    "zoo": handle_zoo_workflow,
    "predict": handle_predict_workflow,
    "vep": handle_vep_workflow,
    "design": handle_design_workflow,
    "scan": handle_scan_workflow,
    "attribution": handle_attribution_workflow,
}