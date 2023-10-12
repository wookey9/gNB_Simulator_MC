from . import cell
import pandas as pd
import os

class gNodeB:
    def __init__(self, N_Cell):
        self.ue_dist = [32,32,32,32,32,32,32,32]
        self.heavy_dist = [0,0,0,0,0,0,0,0]
        self.maxpdu_list = [2,2,2,2,2,2,2,2]

        self.n_cell = N_Cell
        self.cell_list = []
        self.gnb_tput = 0
        for cellId in range(N_Cell):
            self.cell_list.append(cell.Cell(cellId, 66))

        self.episode_size = 20
        self.episode_iter = 0
        self.episode_cnt = 0
        self.running_slot = 100

        if os.path.exists('mobilePhoneActivity/input.pkl'):
            self.mobile_activity = pd.read_pickle('mobilePhoneActivity/input.pkl')
            max_sum = 0
            for i, row in self.mobile_activity.iterrows():
                s = sum(row)
                if s > max_sum:
                    max_sum = s
            self.mobile_activity = (self.mobile_activity * 256 / max_sum).astype(int)
        else:
            self.mobile_activity = pd.DataFrame({})

    def set_epi_size(self,size):
        self.episode_size = size
    def run(self, num_slot):
        if not(len(self.ue_dist) == len(self.heavy_dist) == len(self.maxpdu_list) == len(self.cell_list)):
            return -1

        for cid, cell in enumerate(self.cell_list):
            cell.set_maxpdu(self.maxpdu_list[cid])
            cell.reset_stat()
            cell.release_All()
            for u in range(self.ue_dist[cid]):
                if u < self.heavy_dist[cid]:
                    cell.attach_UE(1)
                else:
                    cell.attach_UE(0)

        self.gnb_tput = 0

        for slot in range(num_slot):
            for cell in self.cell_list:
                self.gnb_tput += cell.schedule(slot)

        return 1

    def get_stat(self):
        cell_tput = []
        cell_rbutil = []
        cell_schedpdu = []
        for cell in self.cell_list:
            rbutil,tput,schedpdu = cell.get_stat(self.running_slot)
            cell_tput.append(tput)
            cell_rbutil.append(rbutil)
            cell_schedpdu.append(schedpdu)

        return self.gnb_tput / 100, cell_tput, cell_rbutil, cell_schedpdu

    def apply_action(self, action):
        minus_cell = abs(action[0] - 8) - 1
        plus_cell = abs(action[1] - 8) - 1

        if minus_cell >= 0:
            self.maxpdu_list[minus_cell] -= 1
        if plus_cell >= 0:
            self.maxpdu_list[plus_cell] += 1

        self.episode_iter += 1

    def observe_state(self):
        self.run(self.running_slot)
        gnb_tput, cell_tput, cell_rb, cell_sched_pdu = self.get_stat()
        state = cell_rb + cell_tput + self.maxpdu_list + self.ue_dist + self.heavy_dist

        done = 0
        if self.episode_iter % self.episode_size == 0:
            done = 1
            self.update_env()
            self.episode_cnt += 1
            self.maxpdu_list = [2,2,2,2,2,2,2,2]
        return state,gnb_tput , done

    def update_env(self):
        if len(self.mobile_activity) > 0:
            self.ue_dist = self.mobile_activity.iloc[self.episode_cnt % len(self.mobile_activity)].values.tolist()
        else:
            self.ue_dist = [32,32,32,32,32,32,32,32]
            self.heavy_dist = [0,0,0,0,0,0,0,0]
