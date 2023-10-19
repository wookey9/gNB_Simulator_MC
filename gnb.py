from . import cell
import pandas as pd
import numpy as np
import os

class gNodeB:
    def __init__(self, N_Cell):
        self.ue_dist = [32,32,32,32,32,32,32,32]
        self.heavy_dist = [0,0,0,0,0,0,0,0]
        self.maxpdu_list = [2,2,2,2,2,2,2,2]
        self.heavy_ratio = [0.3,0.3,0.,0.3,0.3,0.8,0.5,0.5]

        self.n_cell = N_Cell
        self.cell_list = []
        self.gnb_tput = 0
        for cellId in range(N_Cell):
            self.cell_list.append(cell.Cell(cellId, 66))

        self.episode_size = 6
        self.episode_iter = 0
        self.episode_cnt = 0
        self.running_slot = 100

        if os.path.exists('mobilePhoneActivity/input_7267.pkl'):
            self.mobile_activity = pd.read_pickle('mobilePhoneActivity/input_7267.pkl')

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
        self.update_env(self.episode_iter)
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
            self.episode_cnt += 1
            #self.maxpdu_list = [2,2,2,2,2,2,2,2]
        return state,gnb_tput , done

    def update_env(self, step):
        if len(self.mobile_activity) > 0:
            row = self.mobile_activity.iloc[step % len(self.mobile_activity)]
            row = (row * 256 // row.sum()).astype(int)
            self.ue_dist = row.values.tolist()
            self.heavy_dist = list(np.multiply(self.ue_dist,self.heavy_ratio))
        else:
            self.ue_dist = [32,32,32,32,32,32,32,32]
            self.heavy_dist = [0,0,0,0,0,0,0,0]
