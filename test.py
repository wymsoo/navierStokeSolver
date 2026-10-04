from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List


# 1. Enums for fixed types
class LeaveType(Enum):
    ANNUAL = "Annual"
    SICK = "Sick"
    CASUAL = "Casual"


# 2. Base Employee Interface/Class using ABC
class Employee(ABC):
    def __init__(self, emp_id: str, name: str, monthly_salary: float):
        self.emp_id = emp_id
        self.name = name
        self.monthly_salary = monthly_salary
        self.leaves_taken = 0

    @property
    @abstractmethod
    def max_annual_leave(self) -> int:
        """Abstract property: Every subclass MUST define its own quota."""
        pass

    @property
    def remaining_leave(self) -> int:
        return self.max_annual_leave - self.leaves_taken

    def days_present_in_year(self) -> int:
        return 365 - self.leaves_taken


# 3. Specialized Subclasses (Polymorphism)
class SeniorFTE(Employee):
    @property
    def max_annual_leave(self) -> int:
        return 12


class JuniorFTE(Employee):
    @property
    def max_annual_leave(self) -> int:
        return 6


# 4. Leave Management System (Service Layer)
class LeaveSystem:
    def __init__(self, max_company_leaves_per_day: int = 2):
        self.max_leaves_per_day = max_company_leaves_per_day
        self.employees: Dict[str, Employee] = {}
        # Tracks approved leaves per date: {"2026-06-01": ["emp_1", "emp_2"]}
        self.daily_calendar: Dict[str, List[str]] = {}

    def register_employee(self, employee: Employee):
        self.employees[employee.emp_id] = employee

    def apply_leave(self, emp_id: str, leave_date: str, leave_type: LeaveType) -> bool:
        employee = self.employees.get(emp_id)

        # Validation 1: Employee existence
        if not employee:
            print(f"Error: Employee ID {emp_id} not found.")
            return False

        # Validation 2: Individual Quota Check
        if employee.remaining_leave <= 0:
            print(f"Rejected: {employee.name} has no remaining annual leave.")
            return False

        # Validation 3: Company-wide Daily Limit Check
        leaves_on_date = self.daily_calendar.get(leave_date, [])
        if len(leaves_on_date) >= self.max_leaves_per_day:
            print(f"Rejected: Daily leave limit ({self.max_leaves_per_day}) reached for {leave_date}.")
            return False

        # Execution: Approve leave and update state
        employee.leaves_taken += 1
        if leave_date not in self.daily_calendar:
            self.daily_calendar[leave_date] = []
        self.daily_calendar[leave_date].append(emp_id)

        print(f"Approved: {leave_type.value} leave for {employee.name} on {leave_date}.")
        return True


# --- Usage Example ---

system = LeaveSystem(max_company_leaves_per_day=2)

# Instantiate specific employee types
mary = SeniorFTE(emp_id="E001", name="Mary", monthly_salary=3000)
john = JuniorFTE(emp_id="E002", name="John", monthly_salary=1000)
alice = JuniorFTE(emp_id="E003", name="Alice", monthly_salary=1100)

system.register_employee(mary)
system.register_employee(john)
system.register_employee(alice)

# Scenario 1: Applying for leave on the same date
system.apply_leave("E001", "2026-06-01", LeaveType.ANNUAL)  # Approved (1/2)
system.apply_leave("E002", "2026-06-01", LeaveType.SICK)    # Approved (2/2)
system.apply_leave("E003", "2026-06-01", LeaveType.CASUAL)  # Rejected (Max company limit)

# Check presence
print(f"{mary.name} days present: {mary.days_present_in_year()}")  # 364
print(f"{mary.name} remaining leave: {mary.remaining_leave}")      # 11





        


