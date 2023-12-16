

class Specs:

    def __init__(self):
        self.prev_gear = 1.0
        self.current_gear = 1.0
        self.gear_switch_time = 0.0
        '["AT1", "AT2", "AT51", "AT52", "AT53", "AT54", "AT6a", "AT6b", "AT6c"]'
        self.falsified = [-1] * 9 # To show the first run each spec is falsified!
        self.falsification_count = [0]*9 # To show each spec is falsified by how many runs


    def check_falsified_specs(self, speed, RPM, gear, time, run):

        """ This function checks if any of the nine STL specification for
        the Automatic Transmission benchmark from ARCH2019 falsification
        report is falsified by the test cases. If so, it stores """


        '!!! The followingblocks should all be if not elif to test all spects!!!'

        # if  0 <= time <= 20 and speed >= 120: #AT1
        #     self.falsification_count[0] += 1
        #     if self.falsified[0] == -1:
        #         self.falsified[0] = run
        #         # print(f"AT1 falsified. run {run} time {time} speed {speed}")
        #
        # if 0 <= time <= 10 and RPM >= 4750: #AT2
        #     self.falsification_count[1] += 1
        #     if self.falsified[1] == -1:
        #         self.falsified[1] = run
        #         # print(f"AT2 falsified. run {run} time {time} RPM {RPM}")

        if 0 <= time <= 30 and self.current_gear != gear:
            if time - self.gear_switch_time <= 2.5:
                if self.prev_gear != self.current_gear and self.current_gear == 1.0: #AT51
                    self.falsification_count[2] += 1
                    if self.falsified[2] == -1:
                        self.falsified[2] = run
                        print(f"AT51 falsified. run {run} time {time}  self.prev_gear {self.prev_gear}  current_gear {self.current_gear}  gear {gear} time_switch {time - self.gear_switch_time}")
                elif self.prev_gear != self.current_gear and self.current_gear == 2.0: #AT52
                    self.falsification_count[3] += 1
                    if self.falsified[3] == -1:
                        self.falsified[3] = run
                        print(f"AT52 falsified. run {run} time {time}  self.prev_gear {self.prev_gear} current_gear {self.current_gear} gear {gear}  time_switch {time - self.gear_switch_time}")
                elif self.prev_gear != self.current_gear and self.current_gear == 3.0: #AT53
                    self.falsification_count[4] += 1
                    if self.falsified[4] == -1:
                        self.falsified[4] = run
                        print(f"AT53 falsified. run {run} time {time}  self.prev_gear {self.prev_gear} current_gear {self.current_gear} gear {gear} time_switch {time - self.gear_switch_time}")
                elif self.prev_gear != self.current_gear and self.current_gear == 4.0: #AT54
                    self.falsification_count[5] += 1
                    if self.falsified[5] == -1:
                        self.falsified[5] = run
                        print(f"AT54 falsified. run {run} time {time}  self.prev_gear {self.prev_gear} current_gear {self.current_gear} gear {gear} time_switch {time - self.gear_switch_time}  self.gear_switch_time {self.gear_switch_time}")
                else:
                    print(f"Error falsification : Gear switch from {self.current_gear} to {self.current_gear}")

        # if 0 <= time <= 4 and RPM < 3000 and speed >= 35:
        #     self.falsification_count[6] += 1
        #     if self.falsified[6] == -1: #AT6a
        #         self.falsified[6] = run
        #         # print(f"AT6a falsified. run {run} time {time} speed {speed} RPM {RPM}")
        # if 0 <= time <= 8 and RPM < 3000 and speed >= 50:
        #     self.falsification_count[7] += 1
        #     if self.falsified[7] == -1: #AT6b
        #         self.falsified[7] = run
        #         # print(f"AT6b falsified. run {run} time {time} speed {speed} RPM {RPM}")
        # if 0 <= time <= 20 and RPM < 3000 and speed >= 65:
        #     self.falsification_count[8] += 1
        #     if self.falsified[8] == -1: #AT6c
        #         self.falsified[8] = run
        #         # print(f"AT6c falsified. run {run} time {time} speed {speed} RPM {RPM}")

        'Update the gear for next steps'
        if self.current_gear != gear:
            self.prev_gear = self.current_gear
            self.current_gear = gear
            self.gear_switch_time = time


    def print_falsification_result(self):

        # if self.falsified[0] != -1:
        #     print(f"AT1 falsified at run {self.falsified[0]} , successful falsification {self.falsification_count[0]}")
        # if self.falsified[1] != -1:
        #     print(f"AT2 falsified at run {self.falsified[1]} , successful falsification {self.falsification_count[1]}")
        if self.falsified[2] != -1:
            print(f"AT51 falsified at run {self.falsified[2]} , successful falsification {self.falsification_count[2]}")
        if self.falsified[3] != -1:
            print(f"AT52 falsified at run {self.falsified[3]} , successful falsification {self.falsification_count[3]}")
        if self.falsified[4] != -1:
            print(f"AT53 falsified at run {self.falsified[4]} , successful falsification {self.falsification_count[4]}")
        if self.falsified[5] != -1:
            print(f"AT54 falsified at run {self.falsified[5]} , successful falsification {self.falsification_count[5]}")
        # if self.falsified[6] != -1:
        #     print(f"AT6a falsified at run {self.falsified[6]} , successful falsification {self.falsification_count[6]}")
        # if self.falsified[7] != -1:
        #     print(f"AT6b falsified at run {self.falsified[7]} , successful falsification {self.falsification_count[7]}")
        # if self.falsified[8] != -1:
        #     print(f"AT6c falsified at run {self.falsified[8]} , successful falsification {self.falsification_count[8]}")






