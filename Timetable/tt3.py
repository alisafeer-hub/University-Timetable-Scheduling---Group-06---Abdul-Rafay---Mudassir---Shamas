import random
import copy
import csv
import json
import pickle
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from io import StringIO
from datetime import datetime

# Academic Calendar Data
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
    room_type: str
    
@dataclass
class TimeSlot:
    day: str
    hour: int
    duration: int = 2
    
    def __hash__(self):
        return hash((self.day, self.hour))
    
    def __eq__(self, other):
        return self.day == other.day and self.hour == other.hour
    
    def __repr__(self):
        return f"{self.day} {self.hour:02d}:00"

@dataclass
class Assignment:
    course: Course
    room: Room
    timeslot: TimeSlot
    week_start: int
    week_end: int
    
    def __repr__(self):
        return f"{self.course.id} | {self.course.name} | {self.room.id} | {self.timeslot}"


class TrainingMetrics:
    """Track training progress and performance"""
    
    def __init__(self):
        self.generation_history = []
        self.fitness_history = []
        self.conflict_history = []
        self.best_fitness = float('-inf')
        self.training_time = 0
        self.convergence_generation = None
        
    def record(self, generation: int, best_fitness: float, avg_fitness: float, conflicts: int):
        self.generation_history.append(generation)
        self.fitness_history.append({
            'best': best_fitness,
            'average': avg_fitness
        })
        self.conflict_history.append(conflicts)
        
        if best_fitness > self.best_fitness:
            self.best_fitness = best_fitness
            
        if conflicts == 0 and self.convergence_generation is None:
            self.convergence_generation = generation
    
    def summary(self):
        return {
            'total_generations': len(self.generation_history),
            'best_fitness': self.best_fitness,
            'final_conflicts': self.conflict_history[-1] if self.conflict_history else None,
            'convergence_generation': self.convergence_generation,
            'training_time_seconds': self.training_time
        }
    
    def print_summary(self):
        summary = self.summary()
        print("\n" + "=" * 80)
        print("TRAINING SUMMARY")
        print("=" * 80)
        print(f"Total Generations: {summary['total_generations']}")
        print(f"Best Fitness Achieved: {summary['best_fitness']}")
        print(f"Final Conflicts: {summary['final_conflicts']}")
        print(f"Convergence at Generation: {summary['convergence_generation'] or 'Not converged'}")
        print(f"Training Time: {summary['training_time_seconds']:.2f} seconds")
        print("=" * 80)


class UniversityDataGenerator:
    """Generate realistic university data"""
    
    DEPARTMENTS = ["Computer Science", "Mathematics", "Physics", "Chemistry", "Engineering"]
    INSTRUCTORS = [
        "Dr. Smith", "Dr. Johnson", "Dr. Williams", "Dr. Brown", "Dr. Jones",
        "Dr. Garcia", "Dr. Miller", "Dr. Davis", "Dr. Rodriguez", "Dr. Martinez",
        "Dr. Anderson", "Dr. Taylor", "Dr. Thomas", "Dr. Moore", "Dr. Jackson"
    ]
    
    @staticmethod
    def generate_courses(semester: int, num_courses: int = 20) -> List[Course]:
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
        rooms = []
        room_types = {
            "Lecture Hall": (100, 200, 3),
            "Classroom": (30, 50, 6),
            "Lab": (25, 35, 4),
            "Seminar Room": (15, 25, 2)
        }
        
        counter = {"Lecture Hall": 1, "Classroom": 1, "Lab": 1, "Seminar Room": 1}
        
        for room_type, (min_cap, max_cap, count) in room_types.items():
            for _ in range(count):
                capacity = random.randint(min_cap, max_cap)
                room_id = f"{room_type.split()[0][:2].upper()}{counter[room_type]:03d}"
                counter[room_type] += 1
                rooms.append(Room(id=room_id, capacity=capacity, room_type=room_type))
        
        return rooms


class TimetableGeneticAlgorithm:
    """Enhanced Genetic Algorithm with training capabilities"""
    
    def __init__(self, courses: List[Course], rooms: List[Room], 
                 days: List[str], hours: List[int], semester: int,
                 population_size: int = 100, generations: int = 300,
                 mutation_rate: float = 0.15, crossover_rate: float = 0.8,
                 elitism_count: int = 5):
        
        self.courses = courses
        self.rooms = rooms
        self.days = days
        self.hours = hours
        self.semester = semester
        self.timeslots = [TimeSlot(d, h) for d in days for h in hours]
        
        # Hyperparameters
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_count = elitism_count
        
        # Training metrics
        self.metrics = TrainingMetrics()
        self.best_solution = None
        
    def create_individual(self) -> Dict[str, List[Assignment]]:
        """Create a random timetable chromosome"""
        individual = {}
        for course in self.courses:
            individual[course.id] = []
            suitable_rooms = [r for r in self.rooms if r.capacity >= course.students]
            
            used_slots = set()
            for session in range(course.sessions_per_week):
                # Try to avoid same-day scheduling
                attempts = 0
                while attempts < 20:
                    room = random.choice(suitable_rooms) if suitable_rooms else random.choice(self.rooms)
                    timeslot = random.choice(self.timeslots)
                    
                    slot_key = (timeslot.day, timeslot.hour)
                    if slot_key not in used_slots:
                        used_slots.add(slot_key)
                        break
                    attempts += 1
                
                individual[course.id].append(Assignment(
                    course=course,
                    room=room,
                    timeslot=timeslot,
                    week_start=1,
                    week_end=13
                ))
        return individual
    
    def fitness(self, individual: Dict[str, List[Assignment]]) -> Tuple[float, int]:
        """Calculate fitness with detailed conflict counting"""
        conflicts = 0
        all_assignments = []
        for assignments in individual.values():
            all_assignments.extend(assignments)
        
        # Hard constraints
        for i in range(len(all_assignments)):
            a1 = all_assignments[i]
            
            # Room capacity violation
            if a1.room.capacity < a1.course.students:
                conflicts += 5
            
            for j in range(i + 1, len(all_assignments)):
                a2 = all_assignments[j]
                
                # Room conflict (same room, same time)
                if a1.room.id == a2.room.id and a1.timeslot == a2.timeslot:
                    conflicts += 100
                
                # Instructor conflict (same instructor, same time)
                if a1.course.instructor == a2.course.instructor and a1.timeslot == a2.timeslot:
                    conflicts += 100
        
        # Soft constraints (preferences)
        for course_id, assignments in individual.items():
            # Prefer spreading sessions across different days
            days_used = set(a.timeslot.day for a in assignments)
            if len(days_used) < len(assignments):
                conflicts += 5  # Sessions on same day penalty
        
        fitness_score = -conflicts
        return fitness_score, conflicts
    
    def selection(self, population: List[Dict[str, List[Assignment]]],
                 fitness_scores: List[Tuple[float, int]]) -> Dict[str, List[Assignment]]:
        """Tournament selection"""
        tournament_size = 7
        indices = random.sample(range(len(population)), min(tournament_size, len(population)))
        tournament = [(population[i], fitness_scores[i][0]) for i in indices]
        return max(tournament, key=lambda x: x[1])[0]
    
    def crossover(self, parent1: Dict[str, List[Assignment]], 
                 parent2: Dict[str, List[Assignment]]) -> Dict[str, List[Assignment]]:
        """Two-point crossover"""
        if random.random() > self.crossover_rate:
            return copy.deepcopy(parent1)
        
        child = {}
        course_ids = list(parent1.keys())
        point1 = random.randint(0, len(course_ids) - 1)
        point2 = random.randint(point1, len(course_ids))
        
        for i, course_id in enumerate(course_ids):
            if point1 <= i < point2:
                child[course_id] = copy.deepcopy(parent1[course_id])
            else:
                child[course_id] = copy.deepcopy(parent2[course_id])
        
        return child
    
    def mutate(self, individual: Dict[str, List[Assignment]]):
        """Adaptive mutation with multiple strategies"""
        for course_id, assignments in individual.items():
            for i, assignment in enumerate(assignments):
                if random.random() < self.mutation_rate:
                    course = assignment.course
                    
                    # Strategy 1: Change room (70% chance)
                    if random.random() < 0.7:
                        suitable_rooms = [r for r in self.rooms if r.capacity >= course.students]
                        new_room = random.choice(suitable_rooms) if suitable_rooms else random.choice(self.rooms)
                        individual[course_id][i].room = new_room
                    
                    # Strategy 2: Change timeslot (30% chance)
                    else:
                        new_timeslot = random.choice(self.timeslots)
                        individual[course_id][i].timeslot = new_timeslot
    
    def train(self, verbose: bool = True) -> Dict[str, List[Assignment]]:
        """Train the genetic algorithm"""
        start_time = datetime.now()
        
        if verbose:
            print("\n" + "=" * 80)
            print("STARTING GENETIC ALGORITHM TRAINING")
            print("=" * 80)
            print(f"Population Size: {self.population_size}")
            print(f"Generations: {self.generations}")
            print(f"Mutation Rate: {self.mutation_rate}")
            print(f"Crossover Rate: {self.crossover_rate}")
            print(f"Elitism: Top {self.elitism_count} individuals")
            print("=" * 80)
        
        # Initialize population
        population = [self.create_individual() for _ in range(self.population_size)]
        
        best_fitness = float('-inf')
        stagnant_generations = 0
        
        for generation in range(self.generations):
            # Evaluate fitness for entire population
            fitness_scores = [self.fitness(ind) for ind in population]
            
            # Sort by fitness
            sorted_pop = sorted(zip(population, fitness_scores), 
                              key=lambda x: x[1][0], reverse=True)
            population = [ind for ind, _ in sorted_pop]
            fitness_scores = [score for _, score in sorted_pop]
            
            current_best_fitness = fitness_scores[0][0]
            current_conflicts = fitness_scores[0][1]
            avg_fitness = sum(f[0] for f in fitness_scores) / len(fitness_scores)
            
            # Record metrics
            self.metrics.record(generation, current_best_fitness, avg_fitness, current_conflicts)
            
            # Update best solution
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                self.best_solution = copy.deepcopy(population[0])
                stagnant_generations = 0
            else:
                stagnant_generations += 1
            
            # Progress output
            if verbose and (generation % 20 == 0 or current_conflicts == 0):
                print(f"Gen {generation:3d} | Best Fitness: {current_best_fitness:6.0f} | "
                      f"Avg: {avg_fitness:6.0f} | Conflicts: {current_conflicts:3d} | "
                      f"Stagnant: {stagnant_generations}")
            
            # Early stopping
            if current_conflicts == 0:
                if verbose:
                    print(f"\n✓ Perfect solution found at generation {generation}!")
                break
            
            if stagnant_generations > 100:
                if verbose:
                    print(f"\n⚠ Stopping early: stagnant for {stagnant_generations} generations")
                break
            
            # Adaptive mutation rate (increase if stagnant)
            if stagnant_generations > 30:
                self.mutation_rate = min(0.3, self.mutation_rate * 1.1)
            else:
                self.mutation_rate = max(0.1, self.mutation_rate * 0.99)
            
            # Create new population
            new_population = population[:self.elitism_count]  # Elitism
            
            while len(new_population) < self.population_size:
                parent1 = self.selection(population, fitness_scores)
                parent2 = self.selection(population, fitness_scores)
                child = self.crossover(parent1, parent2)
                self.mutate(child)
                new_population.append(child)
            
            population = new_population
        
        # Record training time
        self.metrics.training_time = (datetime.now() - start_time).total_seconds()
        
        if verbose:
            self.metrics.print_summary()
        
        return self.best_solution
    
    def save_model(self, filename: str = "timetable_model.pkl"):
        """Save trained model"""
        model_data = {
            'best_solution': self.best_solution,
            'metrics': self.metrics,
            'hyperparameters': {
                'population_size': self.population_size,
                'generations': self.generations,
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate,
                'elitism_count': self.elitism_count
            },
            'courses': self.courses,
            'rooms': self.rooms,
            'days': self.days,
            'hours': self.hours,
            'semester': self.semester
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n✓ Model saved to {filename}")
    
    @staticmethod
    def load_model(filename: str = "timetable_model.pkl"):
        """Load trained model"""
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        ga = TimetableGeneticAlgorithm(
            courses=model_data['courses'],
            rooms=model_data['rooms'],
            days=model_data['days'],
            hours=model_data['hours'],
            semester=model_data['semester'],
            **model_data['hyperparameters']
        )
        
        ga.best_solution = model_data['best_solution']
        ga.metrics = model_data['metrics']
        
        print(f"\n✓ Model loaded from {filename}")
        return ga


def print_timetable(solution: Dict[str, List[Assignment]], title: str):
    """Print timetable in organized format"""
    print("\n" + "=" * 120)
    print(f"{title}")
    print("=" * 120)
    
    if not solution:
        print("No solution found!")
        return
    
    schedule = defaultdict(list)
    for course_id, assignments in solution.items():
        for assignment in assignments:
            key = (assignment.timeslot.day, assignment.timeslot.hour)
            schedule[key].append(assignment)
    
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    hours = sorted(set(a.timeslot.hour for assignments in solution.values() 
                      for a in assignments))
    
    print(f"\n{'Time':<12}", end="")
    for day in days:
        print(f"{day:<23}", end="")
    print("\n" + "-" * 120)
    
    for hour in hours:
        print(f"{hour:02d}:00-{hour+2:02d}:00", end=" ")
        for day in days:
            key = (day, hour)
            if key in schedule:
                items = schedule[key]
                if items:
                    display = ", ".join([f"{a.course.id}({a.room.id})" for a in items[:2]])
                    print(f"{display[:23]:<23}", end="")
                else:
                    print(" " * 23, end="")
            else:
                print(" " * 23, end="")
        print()


def analyze_solution(solution: Dict[str, List[Assignment]]):
    """Detailed solution analysis"""
    conflicts = []
    all_assignments = []
    for assignments in solution.values():
        all_assignments.extend(assignments)
    
    room_conflicts = 0
    instructor_conflicts = 0
    capacity_violations = 0
    
    for i in range(len(all_assignments)):
        a1 = all_assignments[i]
        
        if a1.room.capacity < a1.course.students:
            capacity_violations += 1
            conflicts.append(f"CAPACITY: {a1.course.id} has {a1.course.students} students "
                           f"but room {a1.room.id} only holds {a1.room.capacity}")
        
        for j in range(i + 1, len(all_assignments)):
            a2 = all_assignments[j]
            
            if a1.room.id == a2.room.id and a1.timeslot == a2.timeslot:
                room_conflicts += 1
                conflicts.append(f"ROOM: {a1.course.id} and {a2.course.id} "
                               f"both in {a1.room.id} at {a1.timeslot}")
            
            if a1.course.instructor == a2.course.instructor and a1.timeslot == a2.timeslot:
                instructor_conflicts += 1
                conflicts.append(f"INSTRUCTOR: {a1.course.instructor} teaching "
                               f"{a1.course.id} and {a2.course.id} at {a1.timeslot}")
    
    print("\n" + "=" * 120)
    print("SOLUTION QUALITY ANALYSIS")
    print("=" * 120)
    print(f"Total Assignments: {len(all_assignments)}")
    print(f"Room Conflicts: {room_conflicts}")
    print(f"Instructor Conflicts: {instructor_conflicts}")
    print(f"Capacity Violations: {capacity_violations}")
    print(f"Total Conflicts: {len(conflicts)}")
    
    if conflicts:
        print(f"\n⚠ Showing first 10 conflicts:")
        for conflict in conflicts[:10]:
            print(f"  • {conflict}")
    else:
        print("\n✓ Perfect solution with zero conflicts!")
    
    return len(conflicts)


# Main execution with training
if __name__ == "__main__":
    print("=" * 120)
    print("UNIVERSITY TIMETABLE AI - TRAINING MODE")
    print("=" * 120)
    
    # Generate data
    print("\n📊 Generating University Data...")
    courses = UniversityDataGenerator.generate_courses(semester=1, num_courses=15)
    rooms = UniversityDataGenerator.generate_rooms(num_rooms=15)
    
    print(f"✓ Generated {len(courses)} courses")
    print(f"✓ Generated {len(rooms)} rooms")
    
    # Define schedule
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    hours = [9, 11, 14, 16]
    
    # Initialize and train model
    print("\n" + "=" * 120)
    print("INITIALIZING GENETIC ALGORITHM")
    print("=" * 120)
    
    ga = TimetableGeneticAlgorithm(
        courses=courses,
        rooms=rooms,
        days=days,
        hours=hours,
        semester=1,
        population_size=150,
        generations=300,
        mutation_rate=0.15,
        crossover_rate=0.85,
        elitism_count=10
    )
    
    # Train the model
    best_solution = ga.train(verbose=True)
    
    # Display results
    print_timetable(best_solution, "OPTIMIZED TIMETABLE (SEMESTER 1)")
    
    # Analyze solution quality
    total_conflicts = analyze_solution(best_solution)
    
    # Save the trained model
    ga.save_model("trained_timetable_model.pkl")
    
    print("\n" + "=" * 120)
    print("TRAINING COMPLETE")
    print("=" * 120)
    print(f"Final Solution Quality: {-ga.fitness(best_solution)[0]} conflicts")
    print("Model has been saved and is ready for deployment!")