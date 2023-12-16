import os
import sys

# explorer_path = '/home/sanaz/Documents/Projects/CoverageGuidedExplorer'
explorer_path = os.path.dirname(os.getcwd())
core_path = explorer_path + '/core'
utility_path = explorer_path + '/utility'
coverage_path = explorer_path + '/coverage'
other_techniques = explorer_path + '/other_techniques'
sys.path.append(explorer_path)
sys.path.append(core_path)
sys.path.append(utility_path)
sys.path.append(coverage_path)
sys.path.append(other_techniques)

import Explorer as explorer
import Random as random


def main():

    print(f"TestSuit:\n")
    # print(f"Usage: \n python3.8 TestSuit.py  method benchmark replay "
    #              f"\n method: explorer, random, breach, staliro "
    #              f"\n benchmark: kinematic_car, ACASXU, automatic_transmission")

    method = input('please enter the test generation method: (explorer, random)]\n')
    benchmark = input('please enter the benchnark: (point_mass, kinematic_car, ACASXU, automatic_transmission)\n')


    if   method == 'explorer':
        explorer.main(benchmark)
    elif method == 'random':
        print("1111")
        random.main(benchmark)
        print("2222")







if __name__ == '__main__':
    main()