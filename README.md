# CoverageExplorer
CoverageExplorer is a tool for test case generation for cyber-physical
systems (CPSs), where the goal is to create tests that maximize a coverage
metric related to continuous states of a CPS.  The tool can systematically
generate test cases for black-box non-linear complex CPSs following a
coverage expansion strategy to spend the test budget generating diverse
test cases representing the CPS behavior [1][2][3].

# Installation and dependencies
To run this program you need python3.8 or higher versions. In addition you
need to have the following tools/programs installed: 

1- AutoKoopman [4] 

2- python packages such as cvxpy

3- Matlab and simulink for automatic transmission benchmark 

# Tool config
You need to place the system you want to work around in the systems
directory. Whatever CPS we choose as the system under test should provide
an interface which gives an instance (object) of the system providing a
step(state, control_input) function which takes in the a system state and
control input and actuate the system and return the next state. We call
this interface preprocessing() and execute it in CoverageExplorer befor
each test case  generation iteration. Please refer to ACASXU,
kinematic_car, and automatic transmission benchmark source code to see how
to define the interface.

You need to set up a config file for each system under test in the config
directory and tune some parameters such as:
        
- run_sim_count: num of simulations for test runs 

- train_sim_count: total num of simulations for model training 

- sim_count_per_train: num of simulations per training iteration 

- traj_per_cluster: percent of trajectories selected from each cluster steps: num of time steps per simulation 

- sd: standard deviation for cps coverage score


# Tool usage:
	
After setting up the config file and making sure about preprocessing()
interface, to run CoverageExplorer change run the following commands:

``` 
cd /directory-to-CoverageExplorer-root-folder/core

python3.8 Explorer system_name
```
	
        
After CoverageExplorer finishes execution to access the state trajectories
and test cases in the form of pickle files run the following command:

```        
ls /directory-to-CoverageExplorer-root-folder/utility/results
````

# Usage Example:

To generate test case, run the followinf commands:

```	
cd ~/Progects/CoverageExplorer/core

python3.8 Explorer kinematic_car

ls ~/Projects/CoverageExplorer/utility/results
```	
	
To plot the state trajectories as output of executing the system with
generated test cases and compute state space coverage run the following
commands:

```
cd /directory-to-CoverageExplorer-root-folder/coverage

python3.8 coverage system_name /directory-to-CoverageExplorer-root-folder/utility/results/system_name_trajectories.pkl
```

# Benchmarks:
	
We have added the following three benchmarks, which you can find them in
the systems directory and read more details about them in [2][3]:
	
- Kinematic car
  
- Neural Network Air-to-air Collision Avoidance (ACASXU)
  
- Automatic Transmission

# References:
	
1- Sheikhi, S., Kim, E., Duggirala, P.S., Bak, S.: Coverage-guided fuzz
testing for cyber-physical systems. In: Proceedings of the 13th
International Conference on Cyber-Physical Systems (2022)
	
2- Sheikhi, S., Bak, S.: Coverage explorer: Coverage-guided test generation
for cyber physical systems. In: Proceedings of the 15th International
Conference on Cyber-Physical Systems (under review). ACM/IEEE (2024)
	   
3- Sheikhi, S., Bak, S.: The CoverageExplorer Tool for CPS Test Case
Generation. In: Proceedings of 16th NASA Formal Methods Symposium (under
review). (NFM 2024)
	
4- E. Lew, A. Hekal, K. Potomkin, N. Kochdumper, B. Hencey, S. Bak and S.
Bogomolov: AutoKoopman: A Toolbox for Automated System Identification via
Koopman Operator Linearization. In: 21st International Symposium on
Automated Technology for Verification and Analysis (ATVA 2023)
