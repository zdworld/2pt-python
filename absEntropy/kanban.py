from rich.console import Console
from rich.panel import Panel

def show_kanban():
    title = '''
                        ██████╗ ██████╗ ████████╗   ██████╗ ██╗   ██╗████████╗██╗  ██╗ ██████╗ ███╗   ██╗
                        ╚════██╗██╔══██╗╚══██╔══╝   ██╔══██╗╚██╗ ██╔╝╚══██╔══╝██║  ██║██╔═══██╗████╗  ██║
                         █████╔╝██████╔╝   ██║█████╗██████╔╝ ╚████╔╝    ██║   ███████║██║   ██║██╔██╗ ██║
                        ██╔═══╝ ██╔═══╝    ██║╚════╝██╔═══╝   ╚██╔╝     ██║   ██╔══██║██║   ██║██║╚██╗██║
                        ███████╗██║        ██║      ██║        ██║      ██║   ██║  ██║╚██████╔╝██║ ╚████║
                        ╚══════╝╚═╝        ╚═╝      ╚═╝        ╚═╝      ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝
                                                                                                    
                                    
                                                Copyright (c) 2024 zyqiao
                                                   All rights Reserved.                                                                                                    
    '''
    console = Console()
    panel = Panel('''
    2PT-python programe is a Python implementation for calculating absolute entropy using the Two-Phase Thermodynamic Model.\n
    2PT-python relias on 2PT-model:
    J. Chem. Phys. 119, 11792-11805 (2003),
    J. Phys. Chem. B 2010, 114, 24, 8191-8198.
                        ''', expand=False, )
    console.print(title)
    console.print(panel)
    console.rule("Task Begin")