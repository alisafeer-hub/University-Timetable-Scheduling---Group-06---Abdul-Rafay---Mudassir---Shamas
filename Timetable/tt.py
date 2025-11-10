import random
import copy
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class Course:
    id: str
    name: str
    instructor: str
    students: int
    duration: int  # in hours

@dataclass
class Room:
    id: str
    capacity: int

@dataclass
class TimeSlot:
    day: str
    hour: int
    
    def __hash__(self):
        return hash((self.day, self.hour))
    
    def __eq__(self, other):
        return self.day == other.day and self.hour == other.hour

@dataclass
class Assignment:
    course: Course
    room: Room
    timeslot: TimeSlot
    
    def __repr__(self):
        return f"{self.course.name} in {self.room.id} at {self.timeslot.day} {self.timeslot.hour}:00"


class TimetableCSP:
    """CSP-based timetable scheduler with heuristics"""
    
    def __init__(self, courses: List[Course], rooms: List[Room], days: List[str], hours: List[int]):
        self.courses = courses
        self.rooms = rooms
        self.days = days
        self.hours = hours
        self.timeslots = [TimeSlot(d, h) for d in days for h in hours]
        
    def get_domain(self, course: Course) -> List[Tuple[Room, TimeSlot]]:
        """Get possible (room, timeslot) pairs for a course"""
        domain = []
        for room in self.rooms:
            if room.capacity >= course.students:
                for timeslot in self.timeslots:
                    domain.append((room, timeslot))
        return domain
    
    def is_consistent(self, assignment: Dict[str, Assignment], course: Course, 
                     room: Room, timeslot: TimeSlot) -> bool:
        """Check if assignment violates constraints"""
        for assigned in assignment.values():
            # Room conflict: same room at same time
            if assigned.room.id == room.id and assigned.timeslot == timeslot:
                return False
            
            # Instructor conflict: same instructor at same time
            if assigned.course.instructor == course.instructor and assigned.timeslot == timeslot:
                return False
        
        return True
    
    def mrv_heuristic(self, unassigned: List[Course], assignment: Dict[str, Assignment]) -> Course:
        """Minimum Remaining Values: choose variable with fewest legal values"""
        min_domain_size = float('inf')
        best_course = None
        
        for course in unassigned:
            domain = self.get_domain(course)
            valid_count = sum(1 for room, ts in domain 
                            if self.is_consistent(assignment, course, room, ts))
            
            if valid_count < min_domain_size:
                min_domain_size = valid_count
                best_course = course
        
        return best_course if best_course else unassigned[0]
    
    def lcv_heuristic(self, course: Course, assignment: Dict[str, Assignment]) -> List[Tuple[Room, TimeSlot]]:
        """Least Constraining Value: order values by impact on other variables"""
        domain = self.get_domain(course)
        valid_domain = [(r, ts) for r, ts in domain 
                       if self.is_consistent(assignment, course, r, ts)]
        
        # Sort by how much each value constrains remaining variables
        def constraint_count(value):
            room, timeslot = value
            count = 0
            for other_course in self.courses:
                if other_course.id not in assignment:
                    other_domain = self.get_domain(other_course)
                    for r, ts in other_domain:
                        if (r.id == room.id and ts == timeslot) or \
                           (other_course.instructor == course.instructor and ts == timeslot):
                            count += 1
            return count
        
        valid_domain.sort(key=constraint_count)
        return valid_domain
    
    def forward_checking(self, assignment: Dict[str, Assignment], 
                        domains: Dict[str, List[Tuple[Room, TimeSlot]]]) -> Dict[str, List[Tuple[Room, TimeSlot]]]:
        """Reduce domains based on current assignment"""
        new_domains = copy.deepcopy(domains)
        
        for course_id, domain in new_domains.items():
            if course_id not in assignment:
                course = next(c for c in self.courses if c.id == course_id)
                new_domains[course_id] = [
                    (r, ts) for r, ts in domain 
                    if self.is_consistent(assignment, course, r, ts)
                ]
        
        return new_domains
    
    def backtrack(self, assignment: Dict[str, Assignment], domains: Dict[str, List[Tuple[Room, TimeSlot]]]) -> Dict[str, Assignment]:
        """Backtracking search with CSP heuristics"""
        if len(assignment) == len(self.courses):
            return assignment
        
        unassigned = [c for c in self.courses if c.id not in assignment]
        course = self.mrv_heuristic(unassigned, assignment)
        
        for room, timeslot in self.lcv_heuristic(course, assignment):
            if self.is_consistent(assignment, course, room, timeslot):
                assignment[course.id] = Assignment(course, room, timeslot)
                
                # Forward checking
                new_domains = self.forward_checking(assignment, domains)
                
                # Check if any domain is empty
                if all(len(d) > 0 for cid, d in new_domains.items() if cid not in assignment):
                    result = self.backtrack(assignment, new_domains)
                    if result:
                        return result
                
                del assignment[course.id]
        
        return None
    
    def solve(self) -> Dict[str, Assignment]:
        """Solve the timetable scheduling problem"""
        domains = {c.id: self.get_domain(c) for c in self.courses}
        return self.backtrack({}, domains)


class GeneticTimetable:
    """Genetic Algorithm for timetable optimization"""
    
    def __init__(self, courses: List[Course], rooms: List[Room], days: List[str], 
                 hours: List[int], population_size: int = 50, generations: int = 100):
        self.courses = courses
        self.rooms = rooms
        self.days = days
        self.hours = hours
        self.timeslots = [TimeSlot(d, h) for d in days for h in hours]
        self.population_size = population_size
        self.generations = generations
    
    def create_individual(self) -> Dict[str, Assignment]:
        """Create a random timetable"""
        individual = {}
        for course in self.courses:
            suitable_rooms = [r for r in self.rooms if r.capacity >= course.students]
            room = random.choice(suitable_rooms) if suitable_rooms else random.choice(self.rooms)
            timeslot = random.choice(self.timeslots)
            individual[course.id] = Assignment(course, room, timeslot)
        return individual
    
    def fitness(self, individual: Dict[str, Assignment]) -> float:
        """Calculate fitness (lower is better, 0 is perfect)"""
        conflicts = 0
        assignments = list(individual.values())
        
        for i in range(len(assignments)):
            for j in range(i + 1, len(assignments)):
                a1, a2 = assignments[i], assignments[j]
                
                # Room conflict
                if a1.room.id == a2.room.id and a1.timeslot == a2.timeslot:
                    conflicts += 10
                
                # Instructor conflict
                if a1.course.instructor == a2.course.instructor and a1.timeslot == a2.timeslot:
                    conflicts += 10
                
                # Room capacity violation
                if a1.room.capacity < a1.course.students:
                    conflicts += 5
        
        return -conflicts  # Negative because we want to maximize
    
    def selection(self, population: List[Dict[str, Assignment]]) -> Dict[str, Assignment]:
        """Tournament selection"""
        tournament_size = 5
        tournament = random.sample(population, tournament_size)
        return max(tournament, key=self.fitness)
    
    def crossover(self, parent1: Dict[str, Assignment], parent2: Dict[str, Assignment]) -> Dict[str, Assignment]:
        """Single-point crossover"""
        child = {}
        courses = list(self.courses)
        crossover_point = random.randint(1, len(courses) - 1)
        
        for i, course in enumerate(courses):
            if i < crossover_point:
                child[course.id] = copy.deepcopy(parent1[course.id])
            else:
                child[course.id] = copy.deepcopy(parent2[course.id])
        
        return child
    
    def mutate(self, individual: Dict[str, Assignment], mutation_rate: float = 0.1):
        """Random mutation"""
        for course_id in individual:
            if random.random() < mutation_rate:
                course = next(c for c in self.courses if c.id == course_id)
                suitable_rooms = [r for r in self.rooms if r.capacity >= course.students]
                room = random.choice(suitable_rooms) if suitable_rooms else random.choice(self.rooms)
                timeslot = random.choice(self.timeslots)
                individual[course_id] = Assignment(course, room, timeslot)
    
    def evolve(self) -> Dict[str, Assignment]:
        """Run genetic algorithm"""
        # Initialize population
        population = [self.create_individual() for _ in range(self.population_size)]
        
        best_individual = None
        best_fitness = float('-inf')
        
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = [(ind, self.fitness(ind)) for ind in population]
            fitness_scores.sort(key=lambda x: x[1], reverse=True)
            
            current_best = fitness_scores[0]
            if current_best[1] > best_fitness:
                best_fitness = current_best[1]
                best_individual = copy.deepcopy(current_best[0])
            
            # Early stopping if perfect solution found
            if best_fitness == 0:
                print(f"Perfect solution found at generation {generation}")
                break
            
            if generation % 10 == 0:
                print(f"Generation {generation}: Best fitness = {best_fitness}")
            
            # Create new population
            new_population = [fitness_scores[0][0]]  # Elitism
            
            while len(new_population) < self.population_size:
                parent1 = self.selection(population)
                parent2 = self.selection(population)
                child = self.crossover(parent1, parent2)
                self.mutate(child)
                new_population.append(child)
            
            population = new_population
        
        return best_individual


# Demo usage
if __name__ == "__main__":
    # Define courses
    courses = [
        Course("CS101", "Data Structures", "Dr. Smith", 30, 2),
        Course("CS102", "Algorithms", "Dr. Johnson", 25, 2),
        Course("CS103", "Database Systems", "Dr. Smith", 35, 2),
        Course("MA101", "Calculus", "Dr. Brown", 40, 2),
        Course("MA102", "Linear Algebra", "Dr. Davis", 30, 2),
        Course("PH101", "Physics", "Dr. Wilson", 45, 2),
    ]
    
    # Define rooms
    rooms = [
        Room("R101", 50),
        Room("R102", 40),
        Room("R103", 30),
        Room("R104", 35),
    ]
    
    # Define time slots
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    hours = [9, 11, 14, 16]
    
    print("=" * 60)
    print("TIMETABLE SCHEDULING USING CSP WITH HEURISTICS")
    print("=" * 60)
    
    csp_solver = TimetableCSP(courses, rooms, days, hours)
    csp_solution = csp_solver.solve()
    
    if csp_solution:
        print("\nCSP Solution found!")
        for assignment in csp_solution.values():
            print(f"  {assignment}")
    else:
        print("\nNo CSP solution found!")
    
    print("\n" + "=" * 60)
    print("TIMETABLE SCHEDULING USING GENETIC ALGORITHM")
    print("=" * 60)
    
    ga_solver = GeneticTimetable(courses, rooms, days, hours, population_size=50, generations=100)
    ga_solution = ga_solver.evolve()
    
    print(f"\nFinal GA Fitness: {ga_solver.fitness(ga_solution)}")
    print("\nGA Solution:")
    for assignment in ga_solution.values():
        print(f"  {assignment}")
    
    # Check conflicts in GA solution
    print("\n" + "=" * 60)
    print("CONFLICT ANALYSIS FOR GA SOLUTION")
    print("=" * 60)
    
    conflicts = []
    assignments = list(ga_solution.values())
    for i in range(len(assignments)):
        for j in range(i + 1, len(assignments)):
            a1, a2 = assignments[i], assignments[j]
            if a1.room.id == a2.room.id and a1.timeslot == a2.timeslot:
                conflicts.append(f"Room conflict: {a1.course.name} and {a2.course.name} in {a1.room.id}")
            if a1.course.instructor == a2.course.instructor and a1.timeslot == a2.timeslot:
                conflicts.append(f"Instructor conflict: {a1.course.instructor} teaching {a1.course.name} and {a2.course.name}")
    
    if conflicts:
        print("\nConflicts found:")
        for conflict in conflicts:
            print(f"  - {conflict}")
    else:
        print("\nNo conflicts found! Perfect timetable generated.")