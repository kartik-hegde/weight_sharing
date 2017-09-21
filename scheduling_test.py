#!/usr/bin/env python3
import numpy as np
import random as rand
import operator

# R = 2
# C = 8
# NUM_VAL = 4    # number of possible weight value
# NUM_PE = 2     # number of PEs
R = 3
C = 512
NUM_VAL = 256    # number of possible weight value
NUM_PE = 16     # number of PEs
ADD_COST = 1
MULT_COST = 20

TOTAL_WORKLOAD = R*R*C*ADD_COST + NUM_VAL*MULT_COST
AVG_WORKLOAD = TOTAL_WORKLOAD / NUM_PE
MAX_WORKLOAD = AVG_WORKLOAD * 1.0   # this value is tunable

AVG_WORKINGSET_SIZE = C / NUM_PE
MAX_WORKINGSET_SIZE = AVG_WORKINGSET_SIZE


class PE:
    def __init__(self, iidx, ikey="WVALUE"):
        self.idx = iidx                     # index of this PE
        self.key = ikey                     # Whether PEs are organized by wvalue or wplane
        self.wvalue_wplane_pairs = []       # list of pairs: (wvalue, wplane)
        self.wvalue_to_wplanes = {}         # dictionary: wvalue -> wplanes that have this wvalue
        self.wplane_to_wvalues = {}         # dictionary: wplane -> wvalues that this wplane has
        self.wvalue_set = set([])           # all wvalues covered
        self.wplane_set = set([])           # all wplanes covered
        self.total_wvalue_occurence = 0     # sum(weight * sum(wplanes * weight_occurence))
        self.workload = 0                   # total computational workload (#ADD * COST_OF_ADD + #MUL * COST_OF_MUL)
        self.workingset_size = 0            # total workingset (#input channels(== #wplanes) needed to fetch into L1)

    ## Member function for adding a pair (wvalue, wplane) to current PE
    def add_wvalue_wplane_pair(self, wvalue_wplane_pair):
        if self.key != "PAIR":
            print("ERROR: PE # ", self.idx, " can only be added works by giving (wvalue, wplane) pairs")
            assert 0
        else:
            self.wvalue_wplane_pairs.append(wvalue_wplane_pair)
            self.wvalue_set = self.wvalue_set.union([wvalue_wplane_pair[0]])
            self.wplane_set = self.wplane_set.union([wvalue_wplane_pair[1]])
            self.total_wvalue_occurence = self.total_wvalue_occurence + 1
            self.workload = self.total_wvalue_occurence * ADD_COST + len(self.wvalue_set) * MULT_COST
            self.workingset_size = len(self.wplane_set)

    ## Member function for adding a wvalue with associated wplanes to current PE
    def add_wvalue(self, wvalue, wplanes, one_filter):
        if self.key != "WVALUE":
            print("ERROR: PE # ", self.idx, " can only be added works by giving wvalues")
            assert 0
        elif wvalue in self.wvalue_to_wplanes:   # this wvalue is already in the bin
            print("ERROR: Weight value ", wvalue, " is already in the PE #", self.idx)
            assert 0
        else:
            self.wvalue_to_wplanes[wvalue] = wplanes
            self.wvalue_set = self.wvalue_set.union([wvalue])
            self.wplane_set = self.wplane_set.union(wplanes)
            for wplane_idx in wplanes:
                for i in range(R):
                    for j in range(R):
                        if one_filter[wplane_idx][i][j] == wvalue:
                            self.total_wvalue_occurence = self.total_wvalue_occurence + 1
            self.workload = self.total_wvalue_occurence * ADD_COST + len(self.wvalue_set) * MULT_COST
            self.workingset_size = len(self.wplane_set)

    ## Member function for adding a wplane with associated wvalues to current PE
    def add_wplane(self, wplane, wvalues, one_filter):
        if self.key != "WPLANE":
            print("ERROR: PE # ", self.idx, " can only be added works by giving wplanes")
            assert 0
        elif wplane in self.wplane_to_wvalues:   # this wplane is already in the bin
            print("ERROR: Weight plane ", wplane, " is already in the PE #", self.idx)
            assert 0
        else:
            self.wplane_to_wvalues[wplane] = wvalues
            self.wplane_set = self.wplane_set.union([wplane])
            self.wvalue_set = self.wvalue_set.union(wvalues)
            self.total_wvalue_occurence = self.total_wvalue_occurence + R*R;
            self.workload = self.total_wvalue_occurence * ADD_COST + len(self.wvalue_set) * MULT_COST
            self.workingset_size = len(self.wplane_set)

    def get_idx(self):
        return self.idx

    def find_wvalue(self, wvalue):
        if wvalue in self.wvalue_set:
            return True
        else:
            return False

    def find_wplane(self, wplane):
        if wplane in self.wplane_set:
            return True
        else:
            return False

    def find_wvalue_wplane_pair(self, wvalue_wplane_pair):
        if self.key != "PAIR":
            print("ERROR: PE # ", self.idx, " cannot be searched by pairs")
            assert 0
        elif wvalue_wplane_pair in self.wvalue_wplane_pairs:
            return True
        else:
            return False

    # def get_wvalues(self):
        # return list(self.wvalue_set)

    # def get_wplanes(self):  # Alias: get_workingset_size
        # return list(self.wplane_set)

    def get_total_wvalue_occurence(self):
        return self.total_wvalue_occurence

    def get_workload(self):
        return self.workload

    def get_workingset_size(self):
        return self.workingset_size

    def get_num_wplanes_overlap(self, wplanes):
        return len(self.wplane_set.intersection(wplanes))

    def get_num_wvalues_overlap(self, wvalues):
        return len(self.wvalue_set.intersection(wvalues))

    def print_PE(self):
        print("==== PE #", self.idx, "====")
        print("wvalues are: ", self.wvalue_set)
        print("wplanes are: ", self.wplane_set)
        print("workload is: ", self.workload)
        print("number of wplanes is: ", len(self.wplane_set))
        print("number of wvalues is: ", len(self.wvalue_set))


################################################
##
##  Helper functions
##
################################################

## Generate a RxRxC filter in which every weight element belongs to (0 ... num_val)
def genOneFilter():
   one_filter = np.random.randint(NUM_VAL, size = (C,R,R))
   return one_filter

## Generate size=NUM_VAL list of lists, each list contains indices of weight planes sharing this weight value
def genWValue2WPlanesList(one_filter):
    wvalue2wplanes_list = []
    for i in range(NUM_VAL):
        wvalue2wplanes_list.append([])
        for c in range(C):
            if i in one_filter[c]:
                wvalue2wplanes_list[i].append(c)
    return wvalue2wplanes_list

## Generate size=C list of lists, each list contains wvalues that locate in this wplane
def genWPlane2WValuesList(one_filter):
    wplane2wvalues_list = []
    for c in range(C):
        wplane2wvalues_list.append([])
        for i in range(NUM_VAL):
            if i in one_filter[c]:
                wplane2wvalues_list[c].append(i)
    return wplane2wvalues_list

## Generate (wvalue, wplane) pair for all wvalues / wplanes, allow repetition for identical (wvalue, wplane) due to existence of one value repeats in one plane
def genWValueWPlanePairs(one_filter):
    wvalue_wplane_pairs = []
    for c in range(C):
        for i in range(R):
            for j in range(R):
                wvalue_wplane_pairs.append((one_filter[c][i][j], c))
    return wvalue_wplane_pairs

# ## Get the union set of wplanes that a list of weights point to
# def getWplanesForWvalues(wvalues, wvalue2wplanes_list):
    # wplane_set = set([])
    # for wvalue in wvalues:
        # wplane_set = wplane_set | set(wvalue2wplane_list[wvalue])
    # return list(wplane_set)


#################################################################################
##
##  Working functions (submodular bload balancing)
##
#################################################################################

## Function:    Schedule_MinWorkload_Greedy
## Requirement: In order to minimize the total workload,
##              we make sure that every wvalue only appears once among all PEs, along with all wplanes associated with it
##              Also trying to minimize the maximum workload by greedy algorithm
## Method:      Initialize by randomly picking NUM_PE wvalue to PEs respectively;
##              for every wvalue, scan from high workload PE to low workload PE;
##              find the PE with workload < MAX_WORKLOAD and with maximum wplane overlap
##              If no overlap OR same amount of overlap across multiple PE, choose PE with lower workload
def Schedule_MinWorkload_Greedy(one_filter, wvalue2wplanes_list):
    print("\n\n")
    print("=========================================")
    print("====== Schedule_MinWorkload_Greedy ======")
    print("=========================================\n")
    # Get a list of all possible weight values
    wvalue_pool = np.arange(NUM_VAL)
    np.random.shuffle(wvalue_pool)

    # Initialize: randomly pick NUM_PE wvalues into each PE
    # Improvement: make sure wvalues have no overlap(or few) on wplanes they have
    pes = []
    for i in range(NUM_PE):
        pes.append(PE(i, "WVALUE"))
        print("Add weight value ", wvalue_pool[0], " to PE #", i)
        pes[i].add_wvalue(wvalue_pool[0],wvalue2wplanes_list[wvalue_pool[0]], one_filter)
        wvalue_pool = np.delete(wvalue_pool, [0])
    print("\n######## Initial State ########")
    for pe in pes:
        pe.print_PE()

    # For every weight value, find the best fit bin:
    print("\n#### Start Adding Unique Weight Values ####")
    for wvalue in wvalue_pool:
        wplanes = wvalue2wplanes_list[wvalue]
        # sort the PEs in terms of workload
        pes.sort(key=operator.attrgetter('workload'), reverse=True)
        max_wplane_overlap = 0
        pe_target = pes[-1]
        for pe in pes:
            if pe.get_workload() <= MAX_WORKLOAD and pe.get_num_wplanes_overlap(wplanes) >= max_wplane_overlap:
                pe_target = pe
                max_wplane_overlap = pe.get_num_wplanes_overlap(wplanes)

        pe_target.add_wvalue(wvalue, wplanes, one_filter)
        print("@@@ Add weight value ", wvalue, wplanes, len(wplanes), " to PE #", pe_target.get_idx())
        print("@@@ Workload of PE #", pe_target.get_idx(), " becomes ", pe_target.get_workload())
        print("@@@ The number of wplane overlap is ", max_wplane_overlap)

    # Output all the PEs
    print("\n######## Final State of PEs ########")
    for pe in pes:
        pe.print_PE()

    return pes

## Function:    Schedule_MinWorkingset_Greedy
## Requirement: In order to minimize working set for each pe
##                  && decrease the duplicate of wplane/input planes within different PEs,
##              we have to split the wplane group assciated with one wvalue into multiple PEs
##              This leads to redundant multiplication for one wvalue.
## Method:      Initialize by randomly picking NUM_PE wplanes into PEs respectively
##              We make sure that the number of wplanes in each PE cannot exceed C/NUM_PE
##              Every time we add one wplane into one PE, the number of extra wvalues added to the PE is minimized
##              If there are multiple candidate PE or no candidate PE, add it to the PE with smallest number of wplanes
def Schedule_MinWorkingset_Greedy(one_filter, wplane2wvalues_list):
    print("\n\n")
    print("===========================================")
    print("====== Schedule_MinWorkingset_Greedy ======")
    print("===========================================\n")
    # Get a list of all possible weight planes
    wplane_pool = np.arange(C)
    np.random.shuffle(wplane_pool)

    # Initialize: randomly pick NUM_PE wplanes into each PE
    # Improvement: make sure wplanes have no overlap(or few) on wvalues they have
    pes = []
    for i in range(NUM_PE):
        pes.append(PE(i, "WPLANE"))
        print("Add weight plane ", wplane_pool[0], " to PE #", i)
        pes[i].add_wplane(wplane_pool[0], wplane2wvalues_list[wplane_pool[0]], one_filter)
        wplane_pool = np.delete(wplane_pool, [0])
    print("\n######## Initial State ########")
    for pe in pes:
        pe.print_PE()

    # For every weight plane, find the best fit bin:
    print("\n#### Start Adding Unique Weight Planes ####")
    for wplane in wplane_pool:
        wvalues = wplane2wvalues_list[wplane]
        # sort the PEs in terms of amount of wplanes
        pes.sort(key=operator.attrgetter('workingset_size'), reverse=True)
        min_wplane_append = C
        pe_target = pes[-1]
        for pe in pes:
            if pe.get_workingset_size() <= MAX_WORKINGSET_SIZE and (len(wvalues) - pe.get_num_wvalues_overlap(wvalues)) <= min_wplane_append:
                pe_target = pe
                min_wplane_append = len(wvalues) - pe.get_num_wvalues_overlap(wvalues)

        pe_target.add_wplane(wplane, wvalues, one_filter)
        print("@@@ Add weight plane ", wplane, wvalues, len(wvalues), " to PE #", pe_target.get_idx())
        print("@@@ Workingset size of PE #", pe_target.get_idx(), " becomes ", pe_target.get_workingset_size())
        print("@@@ The number of wvalue increase is ", min_wplane_append)

    # Output all the PEs
    print("\n######## Final State of PEs ########")
    for pe in pes:
        pe.print_PE()

    return pes

## Function:    Schedule_Hybrid_Greedy
## Requirement: In order to distribute work to each PE in the finest granularity, we have to specify all
##              (wvalue, wplane) pairs (total R*R*C in one filter)
## Method:      Add R*R*C (wvalue, wplane) pairs into NUM_PE PEs:
##              Case I:     Find (wvalue, wplane) in one/more PE: we choose the one with lowest workload
##              Reason:     since the workingset size will not be increased, but workload will be increased
##              Case II:    Find wvalue in one/more PE: we choose one with fewest wplanes
##              Reason:     balance workingset size
##              Case III:   Find wplane in one/more PE: we choose one with lowest workload
##              Reason:     same as I
##              Case IV:    Find no wvalue/wplane in any PE: choose one with fewest wplanes
##              Reason:     same as II
## Note:        Case II and Case III can be switched
def Schedule_Hybrid_Greedy(one_filter, wvalue_wplane_pairs):
    print("\n\n")
    print("====================================")
    print("====== Schedule_Hybrid_Greedy ======")
    print("====================================\n")
    # Get a list of all (wvalue, wplane) pair
    if len(wvalue_wplane_pairs) != R*R*C:
        print("ERROR: number of (wvalue, wplane) pairs must be R*R*C")
        assert 0

    wvalue_wplane_pool = wvalue_wplane_pairs
    rand.shuffle(wvalue_wplane_pool)

    ## Since we take care of the no repeat case, no need to do initialization
    pes = []
    for i in range(NUM_PE):
        pes.append(PE(i, "PAIR"))
    print("\n######## Initial State ########")
    for pe in pes:
        pe.print_PE()

    # For every (wvalue, wplane) pair, find the best bin:
    print("\n#### Start Adding Unique (wvalue, wplane) pair to PEs ####")
    for pair in wvalue_wplane_pool:
        wvalue = pair[0]
        wplane = pair[1]
        # Pass 1 for Case I:
        pe_candidates = []
        for pe in pes:
            if pe.find_wvalue_wplane_pair(pair):
                pe_candidates.append(pe)
        if len(pe_candidates):
            pe_candidates.sort(key=operator.attrgetter('workload'))
            pe_candidates[0].add_wvalue_wplane_pair(pair)
            print("@@@ (DOUBLE_MATCH)", len(pe_candidates), "Add ( wvalue = ", wvalue, ", wplane = ", wplane, ") to PE #", pe_candidates[0].get_idx())
            continue

        # Pass 2.5
        pe_candidates_for_wvalue = []
        pe_candidates_for_wplane = []
        for pe in pes:
            if pe.find_wvalue(wvalue):
                pe_candidates_for_wvalue.append(pe)
            if pe.find_wplane(wplane):
                pe_candidates_for_wplane.append(pe)

        rand.seed()
        if round(rand.random()):
            if len(pe_candidates_for_wplane):
                if len(pe_candidates_for_wplane) == 1:
                    pe_candidates_for_wplane[0].add_wvalue_wplane_pair(pair)
                    print("@@@ (WPLANE_MATCH_ONLY_1)", len(pe_candidates_for_wplane), "Add ( wvalue = ", wvalue, ", wplane = ", wplane, ") to PE #", pe_candidates_for_wplane[0].get_idx())
                    continue
            if len(pe_candidates_for_wvalue):
                if len(pe_candidates_for_wvalue) == 1:
                    pe_candidates_for_wvalue[0].add_wvalue_wplane_pair(pair)
                    print("@@@ (WVALUE_MATCH_ONLY_1)", len(pe_candidates_for_wvalue), "Add ( wvalue = ", wvalue, ", wplane = ", wplane, ") to PE #", pe_candidates_for_wvalue[0].get_idx())
                    continue
        else:
            if len(pe_candidates_for_wvalue):
                if len(pe_candidates_for_wvalue) == 1:
                    pe_candidates_for_wvalue[0].add_wvalue_wplane_pair(pair)
                    print("@@@ (WVALUE_MATCH_ONLY_1)", len(pe_candidates_for_wvalue), "Add ( wvalue = ", wvalue, ", wplane = ", wplane, ") to PE #", pe_candidates_for_wvalue[0].get_idx())
                    continue
            if len(pe_candidates_for_wplane):
                if len(pe_candidates_for_wplane) == 1:
                    pe_candidates_for_wplane[0].add_wvalue_wplane_pair(pair)
                    print("@@@ (WPLANE_MATCH_ONLY_1)", len(pe_candidates_for_wplane), "Add ( wvalue = ", wvalue, ", wplane = ", wplane, ") to PE #", pe_candidates_for_wplane[0].get_idx())
                    continue
        if round(rand.random()):
            if len(pe_candidates_for_wplane):
                pe_candidates_for_wplane.sort(key=operator.attrgetter('workload'))
                pe_candidates_for_wplane[0].add_wvalue_wplane_pair(pair)
                print("@@@ (WPLANE_MATCH_WITH_MANY)", len(pe_candidates_for_wplane), "Add ( wvalue = ", wvalue, ", wplane = ", wplane, ") to PE #", pe_candidates_for_wplane[0].get_idx())
                continue
            if len(pe_candidates_for_wvalue):
                pe_candidates_for_wvalue.sort(key=operator.attrgetter('workingset_size'))
                pe_candidates_for_wvalue[0].add_wvalue_wplane_pair(pair)
                print("@@@ (WVALUE_MATCH_WITH_MANY)", len(pe_candidates_for_wvalue), "Add ( wvalue = ", wvalue, ", wplane = ", wplane, ") to PE #", pe_candidates_for_wvalue[0].get_idx())
                continue
        else:
            if len(pe_candidates_for_wvalue):
                pe_candidates_for_wvalue.sort(key=operator.attrgetter('workingset_size'))
                pe_candidates_for_wvalue[0].add_wvalue_wplane_pair(pair)
                print("@@@ (WVALUE_MATCH_WITH_MANY)", len(pe_candidates_for_wvalue), "Add ( wvalue = ", wvalue, ", wplane = ", wplane, ") to PE #", pe_candidates_for_wvalue[0].get_idx())
                continue
            if len(pe_candidates_for_wplane):
                pe_candidates_for_wplane.sort(key=operator.attrgetter('workload'))
                pe_candidates_for_wplane[0].add_wvalue_wplane_pair(pair)
                print("@@@ (WPLANE_MATCH_WITH_MANY)", len(pe_candidates_for_wplane), "Add ( wvalue = ", wvalue, ", wplane = ", wplane, ") to PE #", pe_candidates_for_wplane[0].get_idx())
                continue

        # # Pass 3 for Case III:
        # assert len(pe_candidates) == 0
        # for pe in pes:
            # if pe.find_wplane(wplane):
                # pe_candidates.append(pe)
        # if len(pe_candidates):
            # pe_candidates.sort(key=operator.attrgetter('workload'))
            # pe_candidates[0].add_wvalue_wplane_pair(pair)
            # print("@@@ (WPLANE_MATCH)", len(pe_candidates), "Add ( wvalue = ", wvalue, ", wplane = ", wplane, ") to PE #", pe_candidates[0].get_idx())
            # continue

        # # Pass 2 for Case II:
        # assert len(pe_candidates) == 0
        # for pe in pes:
            # if pe.find_wvalue(wvalue):
                # pe_candidates.append(pe)
        # if len(pe_candidates):
            # pe_candidates.sort(key=operator.attrgetter('workingset_size'))
            # pe_candidates[0].add_wvalue_wplane_pair(pair)
            # print("@@@ (WVALUE_MATCH)", len(pe_candidates), "Add ( wvalue = ", wvalue, ", wplane = ", wplane, ") to PE #", pe_candidates[0].get_idx())
            # continue

        # Pass 4 for Case IV:
        pe_candidates = pes
        pe_candidates.sort(key=operator.attrgetter('workingset_size'))
        pe_candidates[0].add_wvalue_wplane_pair(pair)
        print("@@@ (NO_MATCH) Add ( wvalue = ", wvalue, ", wplane = ", wplane, ") to PE #", pe_candidates[0].get_idx())

    # Output all the PESs
    print("\n######## Final State of PEs ########")
    for pe in pes:
        pe.print_PE()

    return pes

## Function: Eval (for evaluting pes)
def Eval(pes, name):
    print("\n")
    print("="*len(name) + "=================================")
    print("======== Evaluation - ", name, " ========")
    print("=" * len(name) + "=================================\n")

    # evaluate workload
    max_workload = 0
    tot_workload = 0
    for pe in pes:
        max_workload = max(max_workload, pe.get_workload())
        tot_workload = tot_workload + pe.get_workload()
    print("theoretical maximum total workload: ", (ADD_COST + MULT_COST) * R * R * C)
    print("theoretical minimum total workload: ", TOTAL_WORKLOAD)
    print("theoretical average workload: ", AVG_WORKLOAD)
    print("actual total workload: ", tot_workload)
    print("actual maximum workload: ", max_workload)

    # evaluate wplane dubplicate
    tot_workingset_size = 0
    max_workingset_size = 0
    for pe in pes:
        max_workingset_size = max(max_workingset_size, pe.get_workingset_size())
        tot_workingset_size = tot_workingset_size + pe.get_workingset_size()
    print("theoretical minimum total workingset size (number of wplanes-C): ", C)
    print("actual sum of number of workingset sizes: ", tot_workingset_size)
    print("actual maximum workingset size among all PEs: ", max_workingset_size)



##########################################################################################
##
##  Assumption 1: one integer appears in one wplane at most once
##  Problem 1: Partition NUM_VAL integers into P PElists, with all the wplane associated with this integer;
##              so that (1) the reptitions of wplanes across every list PElist are minimized;
##                          == total number of wplane in PElists are minimized
##                      (2) numbers of wplanes within each PElist are close
##
##########################################################################################

def run():
    # Print Constants:
    print("R = ", R)
    print("C = ", C)
    print("NUM_VAL = ", NUM_VAL)
    print("NUM_PE = ", NUM_PE)
    print("ADD_COST = ", ADD_COST)
    print("MULT_COST = ", MULT_COST)
    print("AVG_WORKLOAD = ", AVG_WORKLOAD)
    print("MAX_WORKLOAD = ", MAX_WORKLOAD)

    # Generate filter
    one_filter = genOneFilter()
    print(one_filter)
    wvalue2wplanes_list = genWValue2WPlanesList(one_filter)
    wplane2wvalues_list = genWPlane2WValuesList(one_filter)
    wvalue_wplane_pairs = genWValueWPlanePairs(one_filter)

    # Scheduling methods
    pes_minworkload = Schedule_MinWorkload_Greedy(one_filter, wvalue2wplanes_list)
    pes_minworkingset = Schedule_MinWorkingset_Greedy(one_filter, wplane2wvalues_list)
    pes_hybrid = Schedule_Hybrid_Greedy(one_filter, wvalue_wplane_pairs)

    # Evalution
    Eval(pes_minworkload, "MinWorkload_Greedy")
    Eval(pes_minworkingset, "MinWorkingset_Greedy")
    Eval(pes_hybrid, "Hybrid_Greedy")

def main():
    # import data
    # get constant
    run()

if __name__ == "__main__":
    main()
