from . import ue

class Cell:
    def __init__(self, cellid, maxRb, maxPdu=2):
        self.ue_cnt = 0
        self.cellId = cellid
        self.max_RB = maxRb
        self.ue_list = []
        self.maxPdu = maxPdu
        self.sch_cnt = 0
        self.rb_utilized = 0
        self.tput = 0

    def attach_UE(self,serviceType):
        self.ue_list.append(ue.UE(len(self.ue_list),serviceType))

    def release_All(self):
        self.ue_list.clear()

    def set_maxpdu(self,maxpdu):
        self.maxPdu = maxpdu

    def schedule(self,slot):
        cell_sch_packetsize = 0
        cell_sch_rbsize = 0
        schpducnt = 0
        searchcnt = 0

        if len(self.ue_list) > 0:
            while schpducnt < self.maxPdu and (schpducnt + searchcnt < len(self.ue_list)):
                ue = self.ue_list[(self.sch_cnt + schpducnt + searchcnt) % len(self.ue_list)]
                if ue.traffic > 0:
                    if ue.aloc_rbcnt + cell_sch_rbsize <= self.max_RB:
                        sched_packetsize, sched_rbsize = ue.allocate(slot)
                        cell_sch_packetsize += sched_packetsize
                        cell_sch_rbsize += sched_rbsize
                        schpducnt += 1
                    else:
                        break
                else:
                    searchcnt += 1

        '''for ue in self.ue_list:
            ue.scheduling_request(slot)'''

        self.sch_cnt += schpducnt

        self.rb_utilized += cell_sch_rbsize
        self.tput += cell_sch_packetsize

        return cell_sch_packetsize

    def get_stat(self, period):
        return self.rb_utilized * 100 / (period * self.max_RB), self.tput / 100, self.sch_cnt / 100

    def reset_stat(self):
        self.rb_utilized = 0
        self.tput = 0
        self.sch_cnt = 0

