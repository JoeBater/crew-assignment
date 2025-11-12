import pandas as pd
import numpy as np
import copy

from datetime import datetime, timedelta, time

class Solver():
    def __init__(self, flightsfilename, crewfilename):

        self.VERBOSE = False

        self.flights = self.get_flights(flightsfilename)
        self.crew = self.get_crew(crewfilename)

        self.assignment = None
        self.crew_state = None

        self.num_destroy_operators = 2
        self.d_assignment = None
        self.d_crew_state = None

        self.num_repair_operators = 2
        self.r_assignment = None
        self.r_crew_state = None

    def get_flights(self, flightsfilename):

        self.flights = pd.read_json(flightsfilename)
        self.flights['dep'] = pd.to_datetime(self.flights['dep']).dt.time
        self.flights['arr'] = pd.to_datetime(self.flights['arr']).dt.time
        return self.flights
    
    def get_crew(self, crewfilename):
        self.crew = pd.read_json(crewfilename)
        return self.crew

    def flight_day(self, fid):
        return int(self.flights[self.flights['id'] == fid]['day'].values[0])

    def flight_origin(self, fid):
        return self.flights[self.flights['id'] == fid]['origin'].values[0]

    def flight_destination(self, fid):
        return self.flights[self.flights['id'] == fid]['dest'].values[0]

    def flight_departure(self, fid):
        return self.flights[self.flights['id'] == fid]['dep'].values[0]

    def flight_arrival(self, fid):
        return self.flights[self.flights['id'] == fid]['arr'].values[0]

    def flight_duration(self, fid):
        return int(self.flights[self.flights['id'] == fid]['duration'].values[0])

    def flight_aircraft_type(self, fid):
        return self.flights[self.flights['id'] == fid]['type'].values[0]



    def initial_assignment(self):
        
        # start with a dictionary comprehension of the flights, assigning None to the "captain" and "first_officer" postions
        self.assignment = {
            fid: {"captain": None, "first_officer": None, "dead_heading": []}
            for fid in self.flights['id']
        }

        # then populate
        for f in range(len(self.flights)):

            fid = self.flights.loc[f, 'id']
            origin = self.flights.loc[f, 'origin']
            aircraft = self.flights.loc[f, 'type']

            eligible_captains = self.crew[
                (self.crew['role'] == 'captain') &
                (self.crew['base'] == origin) &
                (self.crew['qualified'].apply(lambda q: aircraft in q))
            ]['id'].values

            eligible_first_officers = self.crew[
                (self.crew['role'] == 'first_officer') &
                (self.crew['base'] == origin) &
                (self.crew['qualified'].apply(lambda q: aircraft in q))
            ]['id'].values
            #print(eligible_first_officers)

            potential_dead_headers = self.crew[
                (self.crew['base'] == origin) 
            ]['id'].values

            # Assign randomly if eligible crew exist
            if len(eligible_captains) > 0:
                self.assignment[fid]['captain'] = np.random.choice(eligible_captains.tolist())
            if len(eligible_first_officers) > 0:
                self.assignment[fid]['first_officer'] = np.random.choice(eligible_first_officers.tolist())

            # add dead-headers
            for dh in potential_dead_headers:
                if np.random.rand() < 0.002:
                    if dh != self.assignment[fid]['captain'] and dh != self.assignment[fid]['first_officer']:
                        self.assignment[fid]['dead_heading'].append(dh)

        return self.assignment


    # crew_state = {
    #     "C1": {"location": "LHR", "status": "base", "available_at": "06:00"},
    #     "F1": {"location": "AMS", "status": "layover", "available_at": "09:30"},
    #     "C2": {"location": "CDG", "status": "deadheading", "available_at": "08:45"}
    # }

    def initial_crew_state(self):
        
        # create initial blank dictionary with keys for each crew member
        self.crew_state = {}
        for c in range(len(self.crew)):
            if self.crew.loc[c, 'id'] not in self.crew_state:
                self.crew_state[self.crew.loc[c, 'id']] = []

        # now lets populate
        if self.assignment is not None:
            for f in self.assignment:
                #print(assignment[f]['captain'], assignment[f]['first_officer'], assignment[f]['dead_heading'])

                # Extract origin from the specific flight
                flight_row = self.flights[self.flights['id'] == f].iloc[0]
            day = int(flight_row['day'])
            origin = flight_row['origin']
            destination = flight_row['dest']
            depart = flight_row['dep']
            arrive = flight_row['arr']
            #print(f"Flight {f} origin: {origin}")

            if self.assignment[f]['captain'] is not None:
                self.crew_state[self.assignment[f]['captain']].append({"day": day, "flight": f, "role": "captain", "origin": origin, "destination": destination, "depart": depart, "arrive": arrive})
            if self.assignment[f]['first_officer'] is not None:
                self.crew_state[self.assignment[f]['first_officer']].append({"day": day, "flight": f, "role": "first_officer", "origin": origin, "destination": destination, "depart": depart, "arrive": arrive})

            dead_headers = self.assignment[f]['dead_heading']
            if len(dead_headers) > 0:
                #print(f'{len(dead_headers)} dead-headers')
                for dh in dead_headers:
                    self.crew_state[dh].append({"day": day, "flight": f, "role": "dead_heading", "origin": origin, "destination": destination, "depart": depart, "arrive": arrive})

        # ensure crew members flights are sorted
        for c in self.crew_state:
            self.crew_state[c] = sorted(self.crew_state[c], key=lambda x: (x['day'], x['depart']))

        return self.crew_state

    def compute_assignment_cost(self, assignment):
        total_cost = 0
        diagnostics = {
            "unassigned_flights": 0,
            "base_mismatches": 0,
            "qualification_mismatches": 0,
            "deadheading_count": 0,
            "duplicate_roles": 0,
            "crew_usage": {},  # crew_id → count
        }

        for fid, roles in assignment.items():
            flight = self.flights[self.flights['id'] == fid].iloc[0]
            origin = flight['origin']
            aircraft = flight['type']

            for role in ['captain', 'first_officer']:
                crew_id = roles[role]
                if crew_id is None:
                    diagnostics["unassigned_flights"] += 1
                    total_cost += 1000
                    continue

                crew_member = self.crew[self.crew['id'] == crew_id].iloc[0]

                # Base mismatch
                if crew_member['base'] != origin:
                    diagnostics["base_mismatches"] += 1
                    total_cost += 500

                # Qualification mismatch
                if aircraft not in crew_member['qualified']:
                    diagnostics["qualification_mismatches"] += 1
                    total_cost += 1000

                # Track usage
                diagnostics["crew_usage"][crew_id] = diagnostics["crew_usage"].get(crew_id, 0) + 1

            # Role duplication
            if roles['captain'] == roles['first_officer']:
                diagnostics["duplicate_roles"] += 1
                total_cost += 2000

            # Deadheading cost
            dh_count = len(roles['dead_heading'])
            diagnostics["deadheading_count"] += dh_count
            total_cost += dh_count * 100

        return total_cost, diagnostics

    def compute_crew_cost(self, crew_state):
        total_cost = 0
        diagnostics = {}

        for cid, trace in crew_state.items():
            trace = sorted(trace, key=lambda x: (x['day'], x['depart']))  # chronological
            crew_member = self.crew[self.crew['id'] == cid].iloc[0]
            base = crew_member['base']
            last_location = base
            last_arrival = time(0, 0, 0)
            day_flights = {}

            for entry in trace:
                day = entry['day']
                role = entry['role']
                origin = entry['origin']
                destination = entry['destination']
                depart = entry['depart']
                arrive = entry['arrive']

                # Track flights per day
                day_flights[day] = day_flights.get(day, 0) + 1

                # Location mismatch
                if origin != last_location:
                    total_cost += 500  # repositioning penalty

                # Overlap check (simplified)
                if depart < last_arrival:
                    total_cost += 2000  # impossible schedule

                # Deadheading cost
                if role == "dead_heading":
                    total_cost += 100

                last_location = destination
                last_arrival = arrive

            # Fatigue penalty
            for d, count in day_flights.items():
                if count > 3:
                    total_cost += (count - 3) * 300

            # Base return incentive
            if last_location == base:
                total_cost -= 200

            diagnostics[cid] = {
                "flights": len(trace),
                "deadheads": sum(1 for t in trace if t['role'] == "dead_heading"),
                "base_return": last_location == base,
                "fatigue_days": sum(1 for c in day_flights.values() if c > 3),
            }

        return total_cost, diagnostics


    def accept_change(self):
        self.assignment = copy.deepcopy(self.r_assignment)
        self.crew_state = copy.deepcopy(self.r_crew_state)
    
    def destroy(self, num_to_destroy):
        if self.assignment is None or self.crew_state is None:
            raise ValueError("Cannot destroy: assignment or crew_state is None. Call initial_assignment() and initial_crew_state() first.")

        self.d_assignment = copy.deepcopy(self.assignment)
        self.d_crew_state = copy.deepcopy(self.crew_state)

        # destroy
        destroy_operator = np.random.randint(0, self.num_destroy_operators)

        if destroy_operator == 0:
            if self.VERBOSE:
                print("Destroy - using operator 0 => random destroy")

            # random destroy
            flights_to_destroy = np.random.choice(self.flights['id'], num_to_destroy, replace=False)
            

            for ftd in flights_to_destroy:

                affected_crew = set()

                # determine affected crew and remove from assignment
                if self.d_assignment[ftd]['captain'] is not None:
                    affected_crew.add(self.d_assignment[ftd]['captain'])
                if self.d_assignment[ftd]['first_officer'] is not None:
                    affected_crew.add(self.d_assignment[ftd]['first_officer'])
                for ddh in self.d_assignment[ftd]['dead_heading']:
                    affected_crew.add(ddh)
                self.d_assignment[ftd]['captain'] = None
                self.d_assignment[ftd]['first_officer'] = None
                self.d_assignment[ftd]['dead_heading'] = []

                # now remove the destroyed_flight from that crew members crew status
                #print(affected_crew)
                for c in affected_crew:
                    remaining = []
                    if self.d_crew_state[c] is not None:
                        for f in self.d_crew_state[c]:
                            if f['flight'] not in flights_to_destroy:
                                remaining.append(f)
                        self.d_crew_state[c] = remaining

            return self.d_assignment, self.d_crew_state, flights_to_destroy, list(affected_crew)

        if destroy_operator == 1:
            if self.VERBOSE:
                print("Destroy - using operator 1 => target overlapping flights")
            # target overlapping flights

            flights_to_destroy = []
            affected_crew = set()

            for c in self.d_crew_state:
                if len(self.d_crew_state[c]) <= 1:  # ignore if only one or less flight for this crew member
                    continue
                else:
                    remaining_flights = []
                    crew_flights = sorted(self.d_crew_state[c], key=lambda x: (x['day'], x['depart']))
                    
                    for i in range(len(crew_flights)):
                        has_overlap = False
                        current_flight = crew_flights[i]
                        
                        # Check for overlaps with other flights
                        for j in range(len(crew_flights)):
                            if i != j:
                                other_flight = crew_flights[j]
                                # Check if flights are on same day and overlap
                                if (current_flight['day'] == other_flight['day'] and 
                                    current_flight['depart'] < other_flight['arrive'] and 
                                    other_flight['depart'] < current_flight['arrive']):
                                    has_overlap = True
                                    break
                        
                        if has_overlap:
                            # Remove from assignment
                            ftd = current_flight['flight']
                            role = current_flight['role']
                            affected_crew.add(c)
                            flights_to_destroy.append(ftd)
                            
                            if role == 'captain':
                                self.d_assignment[ftd]['captain'] = None
                            elif role == 'first_officer':
                                self.d_assignment[ftd]['first_officer'] = None
                            elif role == 'dead_heading':
                                if c in self.d_assignment[ftd]['dead_heading']:
                                    self.d_assignment[ftd]['dead_heading'].remove(c)
                        else:
                            remaining_flights.append(current_flight)
                    
                    self.d_crew_state[c] = remaining_flights

            return self.d_assignment, self.d_crew_state, flights_to_destroy, list(affected_crew)

          

    def repair(self):
        if self.d_assignment is None or self.d_crew_state is None:
            raise ValueError("Cannot repair: d_assignment or d_crew_state is None. Call destroy() first.")

        self.r_assignment = copy.deepcopy(self.d_assignment)
        self.r_crew_state = copy.deepcopy(self.d_crew_state)

        # repair
        repair_operator = np.random.randint(0, self.num_repair_operators)

        if repair_operator == 0:
            if self.VERBOSE:
                print("Repair - using operator 0 => random repair")

            for fid in self.r_assignment:
                # Check the NEW assignment, not the original
                if self.r_assignment[fid]['captain'] is not None and self.r_assignment[fid]['first_officer'] is not None:
                    continue
                else:
                    #print(new_assignment[fid])
                    aircraft = self.flight_aircraft_type(fid)
                    
                    if self.r_assignment[fid]['captain'] is None:
                        possible_captains = self.crew[
                            (self.crew['role'] == 'captain') &
                            (self.crew['qualified'].apply(lambda q: aircraft in q))
                        ]['id'].values

                        if len(possible_captains) > 0:  # Check if any eligible captains exist
                            repair_captain = np.random.choice(possible_captains.tolist())
                            self.r_assignment[fid]['captain'] = repair_captain

                            # now to insert into the crew-state
                            flight_info = {'day': self.flight_day(fid),
                                'flight': fid,
                                'role': 'captain',
                                'origin': self.flight_origin(fid),
                                'destination': self.flight_destination(fid),
                                'depart': self.flight_departure(fid),
                                'arrive': self.flight_arrival(fid),
                                'duration': self.flight_duration(fid)
                            }

                            self.r_crew_state[repair_captain].append(flight_info)
                            self.r_crew_state[repair_captain].sort(key=lambda f: (f['day'], f['depart']))

                    if self.r_assignment[fid]['first_officer'] is None:
                        possible_first_officers = self.crew[
                            (self.crew['role'] == 'first_officer') &
                            (self.crew['qualified'].apply(lambda q: aircraft in q))
                        ]['id'].values

                        if len(possible_first_officers) > 0:  # Check if any eligible first officers exist
                            repair_first_officer = np.random.choice(possible_first_officers.tolist())
                            self.r_assignment[fid]['first_officer'] = repair_first_officer
                        
                            # now to insert into the crew-state
                            flight_info = {'day': self.flight_day(fid),
                                'flight': fid,
                                'role': 'first_officer',
                                'origin': self.flight_origin(fid),
                                'destination': self.flight_destination(fid),
                                'depart': self.flight_departure(fid),
                                'arrive': self.flight_arrival(fid),
                                'duration': self.flight_duration(fid)
                            }

                            self.r_crew_state[repair_first_officer].append(flight_info)
                            self.r_crew_state[repair_first_officer].sort(key=lambda f: (f['day'], f['depart']))

        if repair_operator == 1:
            if self.VERBOSE:
                print("Repair - using operator 1 => crew location repair")

            for fid in self.r_assignment:
                
                flight = self.flights[self.flights['id'] == fid].iloc[0]
                origin = flight['origin']
                aircraft = flight['type']
                depart = flight['dep']

                # Check the NEW assignment, not the original
                if self.r_assignment[fid]['captain'] is not None and self.r_assignment[fid]['first_officer'] is not None:
                    continue
                else:
                    #print(new_assignment[fid])
                    aircraft = self.flight_aircraft_type(fid)
                    
                    if self.r_assignment[fid]['captain'] is None:

                        possible_captains = self.crew[
                            (self.crew['role'] == 'captain') &
                            (self.crew['qualified'].apply(lambda q: aircraft in q))
                        ]['id'].values

                        if len(possible_captains) > 0:  # Check if any eligible captains exist
                            repair_captain = None
                            for cid in possible_captains:
                                crew_flights = self.r_crew_state.get(cid, []) if self.r_crew_state else []
                                crew_base = self.crew[self.crew['id'] == cid].iloc[0]['base']
                                flight_day = self.flight_day(fid)
                                
                                # Check if crew is available at origin before departure
                                is_available = False
                                current_location = crew_base  # Start at base
                                
                                if crew_flights:
                                    # Sort flights chronologically
                                    sorted_flights = sorted(crew_flights, key=lambda x: (x['day'], x['depart']))
                                    
                                    # Check flights on same day or before
                                    for flight in sorted_flights:
                                        if flight['day'] < flight_day:
                                            current_location = flight['destination']
                                        elif flight['day'] == flight_day and flight['depart'] < depart:
                                            current_location = flight['destination']
                                        elif flight['day'] == flight_day and flight['depart'] >= depart:
                                            break  # Stop at first conflicting flight
                                
                                # Check if crew is at the right location
                                if current_location == origin:
                                    repair_captain = cid
                                    self.r_assignment[fid]['captain'] = cid
                                    break

                            # now to insert into the crew-state if we found a captain
                            if repair_captain is not None:
                                flight_info = {'day': self.flight_day(fid),
                                    'flight': fid,
                                    'role': 'captain',
                                    'origin': self.flight_origin(fid),
                                    'destination': self.flight_destination(fid),
                                    'depart': self.flight_departure(fid),
                                    'arrive': self.flight_arrival(fid),
                                    'duration': self.flight_duration(fid)
                                }
                                self.r_crew_state[repair_captain].append(flight_info)
                                self.r_crew_state[repair_captain].sort(key=lambda f: (f['day'], f['depart']))
                                

                    if self.r_assignment[fid]['first_officer'] is None:
                        possible_first_officers = self.crew[
                            (self.crew['role'] == 'first_officer') &
                            (self.crew['qualified'].apply(lambda q: aircraft in q))
                        ]['id'].values

                        if len(possible_first_officers) > 0:  # Check if any eligible first officers exist
                            repair_first_officer = None
                            for cid in possible_first_officers:
                                crew_flights = self.r_crew_state.get(cid, []) if self.r_crew_state else []
                                crew_base = self.crew[self.crew['id'] == cid].iloc[0]['base']
                                flight_day = self.flight_day(fid)
                                
                                # Check if crew is available at origin before departure
                                is_available = False
                                current_location = crew_base  # Start at base
                                
                                if crew_flights:
                                    # Sort flights chronologically
                                    sorted_flights = sorted(crew_flights, key=lambda x: (x['day'], x['depart']))
                                    
                                    # Check flights on same day or before
                                    for flight in sorted_flights:
                                        if flight['day'] < flight_day:
                                            current_location = flight['destination']
                                        elif flight['day'] == flight_day and flight['depart'] < depart:
                                            current_location = flight['destination']
                                        elif flight['day'] == flight_day and flight['depart'] >= depart:
                                            break  # Stop at first conflicting flight
                                
                                # Check if crew is at the right location
                                if current_location == origin:
                                    repair_first_officer = cid
                                    self.r_assignment[fid]['first_officer'] = cid
                                    break
                        
                            # now to insert into the crew-state if we found a first officer
                            if repair_first_officer is not None:
                                flight_info = {'day': self.flight_day(fid),
                                    'flight': fid,
                                    'role': 'first_officer',
                                    'origin': self.flight_origin(fid),
                                    'destination': self.flight_destination(fid),
                                    'depart': self.flight_departure(fid),
                                    'arrive': self.flight_arrival(fid),
                                    'duration': self.flight_duration(fid)
                                }

                                self.r_crew_state[repair_first_officer].append(flight_info)
                                self.r_crew_state[repair_first_officer].sort(key=lambda f: (f['day'], f['depart']))


        return self.r_assignment, self.r_crew_state


