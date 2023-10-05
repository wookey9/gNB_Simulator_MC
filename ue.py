class UE:
    def __init__(self,uid, type):
        self.uid = uid
        self.service_type = type
        self.sched_cnt = 0
        self.traffic = 0
        self.last_schedslot = 0

        if type == 0:
            self.sr_period = 100
            self.aloc_rbcnt = 8
        else:
            self.sr_period = 100
            self.aloc_rbcnt = 66

    def allocate(self, slot):
        self.sched_cnt += 1
        sched_size = min(self.aloc_rbcnt, self.traffic)
        self.traffic -= sched_size

        self.last_schedslot = slot

        return sched_size, self.aloc_rbcnt

    def scheduling_request(self, slot):
        if self.last_schedslot + self.sr_period < slot:
            if type == 0:
                self.traffic = self.aloc_rbcnt
            else:
                self.traffic = self.aloc_rbcnt * 10

