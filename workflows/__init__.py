from .design import handle_design_workflow
from .info import handle_info_workflow
from .vep import handle_vep_workflow

WORKFLOWS = {
    "info": handle_info_workflow,
    "vep": handle_vep_workflow,
    "design": handle_design_workflow,
}