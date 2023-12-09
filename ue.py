class UE:
    def __init__(self,uid, type, direction):
        self.uid = uid
        self.service_type = type
        self.direction = direction  #0:Uplink, 1:Downlink
        self.sched_cnt = 0
        self.last_schedslot = -1
        self.throughput = 0

        if type == 0:
            self.sr_period = 100
            self.aloc_rbcnt = 4
            self.sch_period = 10
        else:
            self.sr_period = 100
            self.aloc_rbcnt = 66
            self.sch_period = 10
        self.traffic = self.aloc_rbcnt * 10
        #self.scheduling_request(self.sr_period)

    def allocate(self, slot):
        self.sched_cnt += 1
        sched_size = min(self.aloc_rbcnt, self.traffic)
        self.traffic -= sched_size

        self.last_schedslot = slot
        self.throughput += sched_size

        return sched_size, self.aloc_rbcnt

    def is_sch_time(self,slot):
        if self.traffic > 0:
            if self.last_schedslot >= 0:
                return (slot - self.last_schedslot) % self.sch_period == 0
            else:
                return True
        return False

    def scheduling_request(self, slot):
        if self.traffic == 0:
            if (slot - self.last_schedslot) % self.sr_period == 0:
                if self.service_type == 0:
                    self.traffic = self.aloc_rbcnt * 100
                else:
                    self.traffic = self.aloc_rbcnt * 100

