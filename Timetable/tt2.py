import random
import copy
import csv
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
from io import StringIO

# Academic Calendar Data (from uploaded CSV)
ACADEMIC_CALENDAR = """academic_semester,academic_week,calendar_week,date_range,week_start,week_end
1,1,39,2022-09-26/2022-10-02,26/09/2022,10/02/2022
1,2,40,2022-10-03/2022-10-09,03/10/2022,10/09/2022
1,3,41,2022-10-10/2022-10-16,10/10/2022,16/10/2022
1,4,42,2022-10-17/2022-10-23,17/10/2022,23/10/2022
1,5,43,2022-10-24/2022-10-30,24/10/2022,30/10/2022
1,6,44,2022-10-31/2022-11-06,31/10/2022,11/06/2022
1,7,45,2022-11-07/2022-11-13,07/11/2022,13/11/2022
1,8,46,2022-11-14/2022-11-20,14/11/2022,20/11/2022
1,9,47,2022-11-21/2022-11-27,21/11/2022,27/11/2022
1,10,48,2022-11-28/2022-12-04,28/11/2022,12/04/2022
1,11,49,2022-12-05/2022-12-11,05/12/2022,12/11/2022
1,12,50,2022-12-12/2022-12-18,12/12/2022,18/12/2022
1,13,1,2023-01-02/2023-01-08,02/01/2023,01/08/2023
2,1,4,2023-01-23/2023-01-29,23/01/2023,29/01/2023
2,2,5,2023-01-30/2023-02-05,30/01/2023,02/05/2023
2,3,6,2023-02-06/2023-02-12,06/02/2023,02/12/2023
2,4,7,2023-02-13/2023-02-19,13/02/2023,19/02/2023
2,5,8,2023-02-20/2023-02-26,20/02/2023,26/02/2023
2,6,9,2023-02-27/2023-03-05,27/02/2023,03/05/2023
2,7,10,2023-03-06/2023-03-12,06/03/2023,03/12/2023
2,8,11,2023-03-13/2023-03-19,13/03/2023,19/03/2023
2,9,12,2023-03-20/2023-03-26,20/03/2023,26/03/2023
2,10,13,2023-03-27/2023-04-02,27/03/2023,04/02/2023
2,11,17,2023-04-24/2023-04-30,24/04/2023,30/04/2023
2,12,18,2023-05-01/2023-05-07,01/05/2023,05/07/2023
2,13,19,2023-05-08/2023-05-14,08/05/2023,14/05/2023"""

@dataclass
class Course:
    id: str
    name: str
    instructor: str
    department: str
    students: int
    credits: int
    semester: int
    sessions_per_week: int = 2
    
@dataclass
class Room:
    id: str
    capacity: int
    room_type: str  # Lecture Hall, Lab, Seminar Room
    
@dataclass
class TimeSlot:
    day: str
    hour: int
    duration: int = 2  # hours
    
    def __hash__(self):
        return hash((self.day, self.hour))
    
    def __eq__(self, other):
        return self.day == other.day and self.hour == other.hour
    
    def __repr__(self):
        return f"{self.day} {self.hour:02d}:00-{self.hour+self.duration:02d}:00"

@dataclass
class Assignment:
    course: Course
    room: Room
    timeslot: TimeSlot
    week_start: int
    week_end: int
    
    def __repr__(self):
        return f"{self.course.id} | {self.course.name} | {self.course.instructor} | {self.room.id} | {self.timeslot} | Weeks {self.week_start}-{self.week_end}"


class UniversityDataGenerator:
    """Generate realistic university data"""
    
    DEPARTMENTS = ["Computer Science", "Mathematics", "Physics", "Chemistry", "Engineering"]
    INSTRUCTORS = [
        "Dr. Smith", "Dr. Johnson", "Dr. Williams", "Dr. Brown", "Dr. Jones",
        "Dr. Garcia", "Dr. Miller", "Dr. Davis", "Dr. Rodriguez", "Dr. Martinez",
        "Dr. Anderson", "Dr. Taylor", "Dr. Thomas", "Dr. Moore", "Dr. Jackson"
    ]
    
    ROOM_TYPES = {
        "Lecture Hall": (100, 200),
        "Classroom": (30, 50),
        "Lab": (25, 35),
        "Seminar Room": (15, 25)
    }
    
    @staticmethod
    def generate_courses(semester: int, num_courses: int = 20) -> List[Course]:
        """Generate realistic course data"""
        courses = []
        course_templates = [
            ("Data Structures", "Computer Science", 3),
            ("Algorithms", "Computer Science", 3),
            ("Database Systems", "Computer Science", 4),
            ("Operating Systems", "Computer Science", 3),
            ("Computer Networks", "Computer Science", 3),
            ("Software Engineering", "Computer Science", 4),
            ("Calculus I", "Mathematics", 4),
            ("Linear Algebra", "Mathematics", 3),
            ("Discrete Mathematics", "Mathematics", 3),
            ("Probability & Statistics", "Mathematics", 3),
            ("Physics I", "Physics", 4),
            ("Quantum Mechanics", "Physics", 3),
            ("Thermodynamics", "Physics", 3),
            ("Organic Chemistry", "Chemistry", 4),
            ("Inorganic Chemistry", "Chemistry", 3),
            ("Circuit Analysis", "Engineering", 3),
            ("Digital Logic Design", "Engineering", 4),
            ("Signals & Systems", "Engineering", 3),
            ("Control Systems", "Engineering", 3),
            ("Machine Learning", "Computer Science", 4),
        ]
        
        for i, (name, dept, credits) in enumerate(course_templates[:num_courses]):
            course_id = f"{dept[:2].upper()}{101 + i}"
            instructor = random.choice(UniversityDataGenerator.INSTRUCTORS)
            students = random.randint(25, 120)
            sessions = 2 if credits <= 3 else 3
            
            courses.append(Course(
                id=course_id,
                name=name,
                instructor=instructor,
                department=dept,
                students=students,
                credits=credits,
                semester=semester,
                sessions_per_week=sessions
            ))
        
        return courses
    
    @staticmethod
    def generate_rooms(num_rooms: int = 15) -> List[Room]:
        """Generate realistic room data"""
        rooms = []
        room_counter = {"Lecture Hall": 1, "Classroom": 1, "Lab": 1, "Seminar Room": 1}
        
        distribution = {
            "Lecture Hall": 3,
            "Classroom": 6,
            "Lab": 4,
            "Seminar Room": 2
        }
        
        for room_type, count in distribution.items():
            for _ in range(count):
                min_cap, max_cap = UniversityDataGenerator.ROOM_TYPES[room_type]
                capacity = random.randint(min_cap, max_cap)
                room_id = f"{room_type.split()[0][:2].upper()}{room_counter[room_type]:03d}"
                room_counter[room_type] += 1
                
                rooms.append(Room(
                    id=room_id,
                    capacity=capacity,
                    room_type=room_type
                ))
        
        return rooms


class TimetableCSP:
    """CSP-based timetable scheduler with heuristics"""
    
    def __init__(self, courses: List[Course], rooms: List[Room], 
                 days: List[str], hours: List[int], semester: int):
        self.courses = courses
        self.rooms = rooms
        self.days = days
        self.hours = hours
        self.semester = semester
        self.timeslots = [TimeSlot(d, h) for d in days for h in hours]
        
    def get_domain(self, course: Course) -> List[Tuple[Room, TimeSlot]]:
        """Get possible (room, timeslot) pairs for a course"""
        domain = []
        for room in self.rooms:
            if room.capacity >= course.students:
                # Labs prefer lab rooms, large courses prefer lecture halls
                if course.name.endswith("Lab") and room.room_type != "Lab":
                    continue
                if course.students > 80 and room.room_type not in ["Lecture Hall"]:
                    continue
                    
                for timeslot in self.timeslots:
                    domain.append((room, timeslot))
        return domain
    
    def is_consistent(self, assignment: Dict[str, List[Assignment]], 
                     course: Course, room: Room, timeslot: TimeSlot) -> bool:
        """Check if assignment violates constraints"""
        for assigned_list in assignment.values():
            for assigned in assigned_list:
                # Room conflict: same room at same time
                if assigned.room.id == room.id and assigned.timeslot == timeslot:
                    return False
                
                # Instructor conflict: same instructor at same time
                if assigned.course.instructor == course.instructor and assigned.timeslot == timeslot:
                    return False
        
        return True
    
    def mrv_heuristic(self, unassigned: List[Tuple[Course, int]], 
                     assignment: Dict[str, List[Assignment]]) -> Tuple[Course, int]:
        """Minimum Remaining Values heuristic"""
        min_domain_size = float('inf')
        best_item = None
        
        for course, session_num in unassigned:
            domain = self.get_domain(course)
            valid_count = sum(1 for room, ts in domain 
                            if self.is_consistent(assignment, course, room, ts))
            
            if valid_count < min_domain_size:
                min_domain_size = valid_count
                best_item = (course, session_num)
        
        return best_item if best_item else unassigned[0]
    
    def backtrack(self, assignment: Dict[str, List[Assignment]], 
                 unassigned: List[Tuple[Course, int]]) -> bool:
        """Backtracking search"""
        if not unassigned:
            return True
        
        course, session_num = self.mrv_heuristic(unassigned, assignment)
        unassigned.remove((course, session_num))
        
        domain = self.get_domain(course)
        random.shuffle(domain)
        
        for room, timeslot in domain:
            if self.is_consistent(assignment, course, room, timeslot):
                if course.id not in assignment:
                    assignment[course.id] = []
                
                # Assign to full semester
                assignment[course.id].append(Assignment(
                    course=course,
                    room=room,
                    timeslot=timeslot,
                    week_start=1,
                    week_end=13
                ))
                
                if self.backtrack(assignment, unassigned):
                    return True
                
                assignment[course.id].pop()
                if not assignment[course.id]:
                    del assignment[course.id]
        
        unassigned.append((course, session_num))
        return False
    
    def solve(self) -> Dict[str, List[Assignment]]:
        """Solve the timetable scheduling problem"""
        assignment = {}
        unassigned = []
        
        # Create list of all sessions needed
        for course in self.courses:
            for session in range(course.sessions_per_week):
                unassigned.append((course, session))
        
        if self.backtrack(assignment, unassigned):
            return assignment
        return None


class GeneticTimetable:
    """Genetic Algorithm for timetable optimization"""
    
    def __init__(self, courses: List[Course], rooms: List[Room], 
                 days: List[str], hours: List[int], semester: int,
                 population_size: int = 100, generations: int = 200):
        self.courses = courses
        self.rooms = rooms
        self.days = days
        self.hours = hours
        self.semester = semester
        self.timeslots = [TimeSlot(d, h) for d in days for h in hours]
        self.population_size = population_size
        self.generations = generations
    
    def create_individual(self) -> Dict[str, List[Assignment]]:
        """Create a random timetable"""
        individual = {}
        for course in self.courses:
            individual[course.id] = []
            suitable_rooms = [r for r in self.rooms if r.capacity >= course.students]
            
            for session in range(course.sessions_per_week):
                room = random.choice(suitable_rooms) if suitable_rooms else random.choice(self.rooms)
                timeslot = random.choice(self.timeslots)
                individual[course.id].append(Assignment(
                    course=course,
                    room=room,
                    timeslot=timeslot,
                    week_start=1,
                    week_end=13
                ))
        return individual
    
    def fitness(self, individual: Dict[str, List[Assignment]]) -> float:
        """Calculate fitness (higher is better)"""
        conflicts = 0
        all_assignments = []
        for assignments in individual.values():
            all_assignments.extend(assignments)
        
        # Check all pairs for conflicts
        for i in range(len(all_assignments)):
            for j in range(i + 1, len(all_assignments)):
                a1, a2 = all_assignments[i], all_assignments[j]
                
                # Hard constraints (severe penalty)
                if a1.room.id == a2.room.id and a1.timeslot == a2.timeslot:
                    conflicts += 100  # Room conflict
                
                if a1.course.instructor == a2.course.instructor and a1.timeslot == a2.timeslot:
                    conflicts += 100  # Instructor conflict
                
                # Soft constraints (minor penalty)
                if a1.room.capacity < a1.course.students:
                    conflicts += 10  # Room too small
        
        return -conflicts
    
    def selection(self, population: List[Dict[str, List[Assignment]]]) -> Dict[str, List[Assignment]]:
        """Tournament selection"""
        tournament_size = 5
        tournament = random.sample(population, min(tournament_size, len(population)))
        return max(tournament, key=self.fitness)
    
    def crossover(self, parent1: Dict[str, List[Assignment]], 
                 parent2: Dict[str, List[Assignment]]) -> Dict[str, List[Assignment]]:
        """Uniform crossover"""
        child = {}
        for course_id in parent1.keys():
            if random.random() < 0.5:
                child[course_id] = copy.deepcopy(parent1[course_id])
            else:
                child[course_id] = copy.deepcopy(parent2[course_id])
        return child
    
    def mutate(self, individual: Dict[str, List[Assignment]], mutation_rate: float = 0.15):
        """Random mutation"""
        for course_id, assignments in individual.items():
            for i, assignment in enumerate(assignments):
                if random.random() < mutation_rate:
                    course = assignment.course
                    suitable_rooms = [r for r in self.rooms if r.capacity >= course.students]
                    room = random.choice(suitable_rooms) if suitable_rooms else random.choice(self.rooms)
                    timeslot = random.choice(self.timeslots)
                    individual[course_id][i] = Assignment(
                        course=course,
                        room=room,
                        timeslot=timeslot,
                        week_start=1,
                        week_end=13
                    )
    
    def evolve(self) -> Dict[str, List[Assignment]]:
        """Run genetic algorithm"""
        population = [self.create_individual() for _ in range(self.population_size)]
        
        best_individual = None
        best_fitness = float('-inf')
        stagnant_generations = 0
        
        for generation in range(self.generations):
            fitness_scores = [(ind, self.fitness(ind)) for ind in population]
            fitness_scores.sort(key=lambda x: x[1], reverse=True)
            
            current_best_fitness = fitness_scores[0][1]
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_individual = copy.deepcopy(fitness_scores[0][0])
                stagnant_generations = 0
            else:
                stagnant_generations += 1
            
            if best_fitness == 0:
                print(f"✓ Perfect solution found at generation {generation}")
                break
            
            if generation % 20 == 0:
                print(f"Generation {generation}: Best fitness = {best_fitness} | Conflicts = {-best_fitness}")
            
            # Early stopping if stagnant
            if stagnant_generations > 50:
                print(f"Stopping early due to stagnation at generation {generation}")
                break
            
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


def print_timetable(solution: Dict[str, List[Assignment]], title: str):
    """Print timetable in organized format"""
    print("\n" + "=" * 120)
    print(f"{title}")
    print("=" * 120)
    
    if not solution:
        print("No solution found!")
        return
    
    # Organize by day and time
    schedule = defaultdict(list)
    for course_id, assignments in solution.items():
        for assignment in assignments:
            key = (assignment.timeslot.day, assignment.timeslot.hour)
            schedule[key].append(assignment)
    
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    hours = sorted(set(ts.hour for assignments in solution.values() for a in assignments for ts in [a.timeslot]))
    
    print(f"\n{'Time':<12}", end="")
    for day in days:
        print(f"{day:<23}", end="")
    print("\n" + "-" * 120)
    
    for hour in hours:
        print(f"{hour:02d}:00-{hour+2:02d}:00", end=" ")
        for day in days:
            key = (day, hour)
            if key in schedule:
                assignments_at_time = schedule[key]
                if assignments_at_time:
                    a = assignments_at_time[0]
                    print(f"{a.course.id}({a.room.id})".ljust(23), end="")
                else:
                    print(" " * 23, end="")
            else:
                print(" " * 23, end="")
        print()


def analyze_conflicts(solution: Dict[str, List[Assignment]]) -> List[str]:
    """Analyze and report conflicts"""
    conflicts = []
    all_assignments = []
    for assignments in solution.values():
        all_assignments.extend(assignments)
    
    for i in range(len(all_assignments)):
        for j in range(i + 1, len(all_assignments)):
            a1, a2 = all_assignments[i], all_assignments[j]
            
            if a1.room.id == a2.room.id and a1.timeslot == a2.timeslot:
                conflicts.append(f"ROOM CONFLICT: {a1.course.id} and {a2.course.id} both in {a1.room.id} at {a1.timeslot}")
            
            if a1.course.instructor == a2.course.instructor and a1.timeslot == a2.timeslot:
                conflicts.append(f"INSTRUCTOR CONFLICT: {a1.course.instructor} teaching {a1.course.id} and {a2.course.id} at {a1.timeslot}")
    
    return conflicts


# Main execution
if __name__ == "__main__":
    print("=" * 120)
    print("UNIVERSITY TIMETABLE SCHEDULING SYSTEM")
    print("Using Academic Calendar from: 2022-09-26 to 2023-05-14")
    print("=" * 120)
    
    # Generate data
    print("\n📊 Generating University Data...")
    courses_sem1 = UniversityDataGenerator.generate_courses(semester=1, num_courses=15)
    rooms = UniversityDataGenerator.generate_rooms(num_rooms=15)
    
    print(f"✓ Generated {len(courses_sem1)} courses for Semester 1")
    print(f"✓ Generated {len(rooms)} rooms")
    
    # Print course details
    print("\n" + "=" * 120)
    print("COURSE DETAILS")
    print("=" * 120)
    print(f"{'ID':<8} {'Course Name':<30} {'Instructor':<15} {'Students':<10} {'Credits':<8} {'Sessions/Week'}")
    print("-" * 120)
    for course in courses_sem1:
        print(f"{course.id:<8} {course.name:<30} {course.instructor:<15} {course.students:<10} {course.credits:<8} {course.sessions_per_week}")
    
    # Define time configuration
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    hours = [9, 11, 14, 16]  # 9-11, 11-13, 14-16, 16-18
    
    # Try CSP first
    print("\n" + "=" * 120)
    print("ATTEMPTING CSP SOLUTION WITH HEURISTICS")
    print("=" * 120)
    
    csp_solver = TimetableCSP(courses_sem1, rooms, days, hours, semester=1)
    csp_solution = csp_solver.solve()
    
    if csp_solution:
        print("\n✓ CSP Solution found!")
        print_timetable(csp_solution, "SEMESTER 1 TIMETABLE (CSP Solution)")
        conflicts = analyze_conflicts(csp_solution)
        if conflicts:
            print("\n⚠ Conflicts detected:")
            for conflict in conflicts[:10]:
                print(f"  {conflict}")
        else:
            print("\n✓ No conflicts! Perfect timetable generated by CSP.")
    else:
        print("\n✗ CSP could not find a solution. Trying Genetic Algorithm...")
        
        # Use GA if CSP fails
        print("\n" + "=" * 120)
        print("RUNNING GENETIC ALGORITHM")
        print("=" * 120)
        
        ga_solver = GeneticTimetable(courses_sem1, rooms, days, hours, semester=1, 
                                    population_size=100, generations=200)
        ga_solution = ga_solver.evolve()
        
        print(f"\nFinal GA Fitness Score: {ga_solver.fitness(ga_solution)}")
        print_timetable(ga_solution, "SEMESTER 1 TIMETABLE (GA Solution)")
        
        conflicts = analyze_conflicts(ga_solution)
        print("\n" + "=" * 120)
        print("CONFLICT ANALYSIS")
        print("=" * 120)
        if conflicts:
            print(f"\n⚠ {len(conflicts)} conflicts found:")
            for conflict in conflicts[:15]:
                print(f"  {conflict}")
        else:
            print("\n✓ No conflicts! Perfect timetable generated.")