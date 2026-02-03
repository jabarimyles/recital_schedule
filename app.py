#!/usr/bin/env python3
"""
Ballet Recital Scheduler - Streamlit Web App
Upload your class/dancer data and get an optimized schedule using PuLP
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Set
import pulp
import io

# Page config
st.set_page_config(
    page_title="Ballet Recital Scheduler",
    page_icon="🩰",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #E91E63;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)


class DanceRecitalSchedulerFromFile:
    """
    Dance Recital Scheduling System
    Uses PuLP for optimization
    """
    
    def __init__(self):
        self.data_loaded = False
    
    def load_from_dataframes(self, config: dict, classes_df: pd.DataFrame, 
                             dancers_df: pd.DataFrame, enrollments_df: pd.DataFrame,
                             siblings_df: pd.DataFrame, constraints_df: pd.DataFrame):
        """Load all data from dataframes"""
        
        # Configuration
        self.num_days = int(config.get('num_days', 3))
        self.min_change_time = int(config.get('min_change_time', 30))
        self.max_duration_per_day = int(config.get('max_duration_per_day', 240))
        self.setup_buffer = int(config.get('setup_buffer', 5))
        
        start_str = str(config.get('start_time', '14:00'))
        try:
            self.start_time = datetime.strptime(start_str, "%H:%M")
        except:
            self.start_time = datetime.strptime("14:00", "%H:%M")
        
        # Classes
        self.classes = {}
        for _, row in classes_df.iterrows():
            self.classes[row['Class_Name']] = {
                'duration': int(row['Duration_Minutes']),
                'avg_age': float(row['Average_Age']),
                'num_dancers': int(row['Number_of_Dancers'])
            }
        
        # Dancers
        self.dancers = {}
        for _, row in dancers_df.iterrows():
            self.dancers[int(row['Dancer_ID'])] = {
                'name': row['Dancer_Name'],
                'age': int(row['Age']) if pd.notna(row['Age']) else None
            }
        
        # Enrollments
        self.dancer_classes = {}
        for _, row in enrollments_df.iterrows():
            dancer_id = int(row['Dancer_ID'])
            class_name = row['Class_Name']
            
            if dancer_id not in self.dancer_classes:
                self.dancer_classes[dancer_id] = []
            self.dancer_classes[dancer_id].append(class_name)
        
        # Update class dancer counts based on actual enrollments
        for class_name in self.classes:
            actual_count = sum(1 for dancer_id, classes in self.dancer_classes.items() 
                             if class_name in classes)
            if actual_count > 0:
                self.classes[class_name]['num_dancers'] = actual_count
        
        # Siblings
        self.siblings = []
        if not siblings_df.empty and 'Sibling_Group' in siblings_df.columns:
            for group_id in siblings_df['Sibling_Group'].unique():
                group_dancers = siblings_df[siblings_df['Sibling_Group'] == group_id]['Dancer_ID'].tolist()
                if len(group_dancers) > 1:
                    self.siblings.append(set(int(d) for d in group_dancers))
        
        # Constraints
        self.must_same_day = []
        self.must_different_day = []
        if not constraints_df.empty:
            for _, row in constraints_df.iterrows():
                constraint_type = str(row['Constraint_Type']).lower()
                class1 = row['Class1']
                class2 = row['Class2']
                
                if constraint_type == 'same_day':
                    self.must_same_day.append((class1, class2))
                elif constraint_type == 'different_day':
                    self.must_different_day.append((class1, class2))
        
        self.data_loaded = True
        return self.get_data_summary()
    
    def get_data_summary(self) -> dict:
        """Return summary of loaded data"""
        multi_class = sum(1 for classes in self.dancer_classes.values() if len(classes) > 1)
        return {
            'classes': len(self.classes),
            'dancers': len(self.dancers),
            'sibling_groups': len(self.siblings),
            'days': self.num_days,
            'enrollments': sum(len(classes) for classes in self.dancer_classes.values()),
            'multi_class_dancers': multi_class,
            'same_day_constraints': len(self.must_same_day),
            'diff_day_constraints': len(self.must_different_day)
        }
    
    def solve_stage1_day_assignment(self) -> Dict[str, int]:
        """Stage 1: Assign classes to days using PuLP"""
        
        if not self.data_loaded:
            raise ValueError("No data loaded.")
        
        model = pulp.LpProblem("Day_Assignment", pulp.LpMinimize)
        
        # Decision variables: x[class, day] = 1 if class assigned to day
        x = pulp.LpVariable.dicts("assign", 
                                  [(c, d) for c in self.classes for d in range(1, self.num_days + 1)],
                                  cat='Binary')
        
        # Constraint 1: Each class assigned to exactly one day
        for class_name in self.classes:
            model += pulp.lpSum([x[class_name, d] for d in range(1, self.num_days + 1)]) == 1
        
        # Constraint 2: Sibling constraints (soft - minimize violations)
        sibling_violations = []
        for idx, sibling_set in enumerate(self.siblings):
            sibling_classes = set()
            for dancer_id in sibling_set:
                if dancer_id in self.dancer_classes:
                    sibling_classes.update(self.dancer_classes[dancer_id])
            
            if len(sibling_classes) > 1:
                violation = pulp.LpVariable(f"sibling_violation_{idx}", lowBound=0)
                sibling_violations.append(violation)
                
                for day in range(1, self.num_days + 1):
                    class_list = list(sibling_classes)
                    if len(class_list) >= 2 and all(c in self.classes for c in class_list):
                        for i in range(len(class_list) - 1):
                            model += x[class_list[i], day] - x[class_list[i+1], day] <= violation
                            model += x[class_list[i+1], day] - x[class_list[i], day] <= violation
        
        # Constraint 3: Must same day
        for class1, class2 in self.must_same_day:
            if class1 in self.classes and class2 in self.classes:
                for day in range(1, self.num_days + 1):
                    model += x[class1, day] == x[class2, day]
        
        # Constraint 4: Must different day
        for class1, class2 in self.must_different_day:
            if class1 in self.classes and class2 in self.classes:
                for day in range(1, self.num_days + 1):
                    model += x[class1, day] + x[class2, day] <= 1
        
        # Constraint 5: Balance performances per day
        max_perf_diff = pulp.LpVariable("max_perf_diff", lowBound=0)
        perf_per_day = []
        
        for day in range(1, self.num_days + 1):
            perf_count = pulp.lpSum([x[c, day] for c in self.classes])
            perf_per_day.append(perf_count)
        
        for i in range(len(perf_per_day)):
            for j in range(i+1, len(perf_per_day)):
                model += perf_per_day[i] - perf_per_day[j] <= max_perf_diff
                model += perf_per_day[j] - perf_per_day[i] <= max_perf_diff
        
        # Objective: minimize sibling violations and imbalance
        model += 100 * pulp.lpSum(sibling_violations) + 10 * max_perf_diff
        
        # Solve
        model.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # Extract solution
        day_assignment = {}
        for class_name in self.classes:
            for day in range(1, self.num_days + 1):
                if pulp.value(x[class_name, day]) is not None and pulp.value(x[class_name, day]) > 0.5:
                    day_assignment[class_name] = day
                    break
        
        # Ensure all classes are assigned (fallback)
        for class_name in self.classes:
            if class_name not in day_assignment:
                day_counts = {d: sum(1 for c, day in day_assignment.items() if day == d)
                             for d in range(1, self.num_days + 1)}
                day_assignment[class_name] = min(day_counts, key=day_counts.get)
        
        return day_assignment
    
    def solve_stage2_time_scheduling(self, day_assignment: Dict[str, int]) -> Dict:
        """Stage 2: Schedule time slots within each day"""
        
        schedule = {}
        
        for day in range(1, self.num_days + 1):
            day_classes = [c for c, d in day_assignment.items() if d == day]
            
            if not day_classes:
                continue
            
            # Sort by age for natural flow
            day_classes.sort(key=lambda c: self.classes[c]['avg_age'])
            
            current_time = self.start_time
            day_schedule = []
            
            for class_name in day_classes:
                duration = self.classes[class_name]['duration']
                
                day_schedule.append({
                    'class': class_name,
                    'start_time': current_time,
                    'end_time': current_time + timedelta(minutes=duration),
                    'duration': duration,
                    'dancers': self.classes[class_name]['num_dancers'],
                    'avg_age': self.classes[class_name]['avg_age']
                })
                
                current_time += timedelta(minutes=duration + self.setup_buffer)
            
            schedule[day] = day_schedule
        
        return schedule
    
    def run_optimization(self):
        """Run complete optimization pipeline"""
        
        if not self.data_loaded:
            raise ValueError("No data loaded.")
        
        # Stage 1: Day assignment
        day_assignment = self.solve_stage1_day_assignment()
        
        # Stage 2: Time scheduling
        schedule = self.solve_stage2_time_scheduling(day_assignment)
        
        # Convert to DataFrame
        data = []
        for day, day_schedule in schedule.items():
            for item in day_schedule:
                data.append({
                    'Day': day,
                    'Class': item['class'],
                    'Start_Time': item['start_time'].strftime('%H:%M'),
                    'End_Time': item['end_time'].strftime('%H:%M'),
                    'Duration_Minutes': item['duration'],
                    'Number_of_Dancers': item['dancers'],
                    'Average_Age': item['avg_age']
                })
        
        schedule_df = pd.DataFrame(data)
        
        return day_assignment, schedule, schedule_df


def load_excel_data(uploaded_file):
    """Load and validate data from Excel file"""
    
    try:
        xlsx = pd.ExcelFile(uploaded_file)
        
        # Load each sheet
        config_df = pd.read_excel(xlsx, sheet_name='Configuration')
        classes_df = pd.read_excel(xlsx, sheet_name='Classes')
        dancers_df = pd.read_excel(xlsx, sheet_name='Dancers')
        enrollments_df = pd.read_excel(xlsx, sheet_name='Enrollments')
        
        # Optional sheets
        try:
            siblings_df = pd.read_excel(xlsx, sheet_name='Siblings')
        except:
            siblings_df = pd.DataFrame(columns=['Sibling_Group', 'Dancer_ID'])
        
        try:
            constraints_df = pd.read_excel(xlsx, sheet_name='Constraints')
        except:
            constraints_df = pd.DataFrame(columns=['Constraint_Type', 'Class1', 'Class2', 'Reason'])
        
        # Convert config to dict
        config = dict(zip(config_df['Parameter'], config_df['Value']))
        
        return config, classes_df, dancers_df, enrollments_df, siblings_df, constraints_df, None
        
    except Exception as e:
        return None, None, None, None, None, None, str(e)


def create_sample_template():
    """Create a sample Excel template for download"""
    
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Configuration
        config_df = pd.DataFrame({
            'Parameter': ['num_days', 'min_change_time', 'start_time', 'max_duration_per_day', 'setup_buffer'],
            'Value': [3, 30, '14:00', 240, 5],
            'Description': [
                'Number of recital days',
                'Minimum minutes between performances for costume changes',
                'Start time for performances (HH:MM format)',
                'Maximum duration per day in minutes',
                'Minutes between performances for stage setup'
            ]
        })
        config_df.to_excel(writer, sheet_name='Configuration', index=False)
        
        # Classes
        classes_df = pd.DataFrame({
            'Class_Name': [
                'Ballet_1', 'Ballet_2', 'Ballet_3', 'Ballet_4',
                'Jazz_1', 'Jazz_2', 'Jazz_3',
                'Tap_1', 'Tap_2',
                'Contemporary_1', 'Contemporary_2',
                'HipHop_1', 'HipHop_2',
                'Lyrical_1', 'Lyrical_2'
            ],
            'Duration_Minutes': [3, 4, 5, 6, 3, 4, 4, 3, 4, 5, 5, 3, 4, 5, 5],
            'Average_Age': [5, 7, 10, 15, 6, 8, 10, 7, 9, 11, 13, 9, 12, 14, 16],
            'Number_of_Dancers': [12, 15, 12, 8, 14, 12, 10, 11, 13, 8, 10, 16, 14, 9, 11]
        })
        classes_df.to_excel(writer, sheet_name='Classes', index=False)
        
        # Dancers
        dancers_df = pd.DataFrame({
            'Dancer_ID': list(range(1, 101)),
            'Dancer_Name': [f'Dancer_{i}' for i in range(1, 101)],
            'Age': [5 + (i % 12) for i in range(1, 101)]
        })
        dancers_df.to_excel(writer, sheet_name='Dancers', index=False)
        
        # Enrollments
        enrollments = []
        class_list = classes_df['Class_Name'].tolist()
        dancer_id = 1
        for class_name in class_list:
            class_info = classes_df[classes_df['Class_Name'] == class_name].iloc[0]
            num_dancers = int(class_info['Number_of_Dancers'] * 0.8)
            for _ in range(num_dancers):
                if dancer_id <= 100:
                    enrollments.append({'Dancer_ID': dancer_id, 'Class_Name': class_name})
                    dancer_id += 1
        # Multi-class dancers
        for i in range(20):
            dancer = 10 + i * 3
            if dancer <= 100:
                second_class = class_list[(i + 5) % len(class_list)]
                enrollments.append({'Dancer_ID': dancer, 'Class_Name': second_class})
        enrollments_df = pd.DataFrame(enrollments)
        enrollments_df.to_excel(writer, sheet_name='Enrollments', index=False)
        
        # Siblings
        siblings_df = pd.DataFrame({
            'Sibling_Group': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5],
            'Dancer_ID': [1, 2, 5, 6, 10, 11, 15, 16, 20, 21, 22]
        })
        siblings_df.to_excel(writer, sheet_name='Siblings', index=False)
        
        # Constraints
        constraints_df = pd.DataFrame({
            'Constraint_Type': ['same_day', 'same_day', 'different_day', 'different_day'],
            'Class1': ['Ballet_1', 'Lyrical_1', 'Contemporary_1', 'HipHop_1'],
            'Class2': ['Ballet_2', 'Lyrical_2', 'Contemporary_2', 'HipHop_2'],
            'Reason': ['Share some students', 'Advanced dancers in both', 
                      'Spread contemporary performances', 'Avoid scheduling conflicts']
        })
        constraints_df.to_excel(writer, sheet_name='Constraints', index=False)
    
    output.seek(0)
    return output


def main():
    """Main Streamlit app"""
    
    st.markdown('<h1 class="main-header">🩰 Ballet Recital Scheduler</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload your class data and get an optimized performance schedule</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Instructions")
        st.markdown("""
        **Step 1:** Download the template
        
        **Step 2:** Fill in your data:
        - **Configuration**: Days, times, etc.
        - **Classes**: All performances
        - **Dancers**: All participants
        - **Enrollments**: Who's in which class
        - **Siblings**: Group siblings
        - **Constraints**: Same/different day rules
        
        **Step 3:** Upload & generate!
        """)
        
        st.divider()
        st.header("📥 Get Template")
        template_data = create_sample_template()
        st.download_button(
            label="Download Excel Template",
            data=template_data,
            file_name="dance_recital_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    # Main area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📤 Upload Your Data")
        uploaded_file = st.file_uploader(
            "Upload your completed Excel file",
            type=['xlsx'],
            help="Use the template from the sidebar"
        )
    
    with col2:
        st.header("⚙️ Options")
        show_details = st.checkbox("Show data preview", value=False)
    
    if uploaded_file is not None:
        # Load data
        config, classes_df, dancers_df, enrollments_df, siblings_df, constraints_df, error = load_excel_data(uploaded_file)
        
        if error:
            st.error(f"Error loading file: {error}")
            return
        
        st.success("✅ File loaded successfully!")
        
        # Preview
        if show_details:
            with st.expander("📊 Data Preview", expanded=True):
                tab1, tab2, tab3, tab4 = st.tabs(["Classes", "Dancers", "Enrollments", "Constraints"])
                with tab1:
                    st.dataframe(classes_df, use_container_width=True)
                with tab2:
                    st.dataframe(dancers_df.head(20), use_container_width=True)
                with tab3:
                    st.dataframe(enrollments_df.head(20), use_container_width=True)
                with tab4:
                    st.dataframe(constraints_df, use_container_width=True)
        
        # Summary metrics
        st.subheader("📈 Data Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Classes", len(classes_df))
        with col2:
            st.metric("Dancers", len(dancers_df))
        with col3:
            st.metric("Days", config.get('num_days', 3))
        with col4:
            st.metric("Total Duration", f"{classes_df['Duration_Minutes'].sum()} min")
        
        st.divider()
        
        # Generate button
        if st.button("🎭 Generate Schedule", type="primary", use_container_width=True):
            
            with st.spinner("Running optimization... This may take a moment."):
                try:
                    # Create scheduler
                    scheduler = DanceRecitalSchedulerFromFile()
                    summary = scheduler.load_from_dataframes(
                        config, classes_df, dancers_df,
                        enrollments_df, siblings_df, constraints_df
                    )
                    
                    # Run optimization
                    day_assignment, schedule, schedule_df = scheduler.run_optimization()
                    
                    if schedule_df.empty:
                        st.error("Could not generate schedule. Please check your data.")
                        return
                    
                    st.success("🎉 Schedule generated successfully!")
                    
                    # Display by day
                    st.header("📅 Your Recital Schedule")
                    
                    for day in sorted(schedule_df['Day'].unique()):
                        day_data = schedule_df[schedule_df['Day'] == day].copy()
                        day_duration = day_data['Duration_Minutes'].sum()
                        
                        with st.expander(f"Day {day}: {len(day_data)} performances ({day_duration} min)", expanded=True):
                            display_df = day_data[['Start_Time', 'End_Time', 'Class', 
                                                   'Duration_Minutes', 'Number_of_Dancers', 'Average_Age']].copy()
                            display_df.columns = ['Start', 'End', 'Class', 'Duration', '# Dancers', 'Avg Age']
                            st.dataframe(display_df, use_container_width=True, hide_index=True)
                    
                    # Downloads
                    st.divider()
                    st.subheader("📥 Download Schedule")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        csv_data = schedule_df.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv_data,
                            file_name="recital_schedule.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    with col2:
                        excel_buffer = io.BytesIO()
                        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                            schedule_df.to_excel(writer, sheet_name='Schedule', index=False)
                        excel_buffer.seek(0)
                        st.download_button(
                            label="Download Excel",
                            data=excel_buffer,
                            file_name="recital_schedule.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
                    
                except Exception as e:
                    st.error(f"Error generating schedule: {str(e)}")
                    st.exception(e)
    
    else:
        st.info("👆 Upload your Excel file to get started, or download the template from the sidebar.")
        
        with st.expander("ℹ️ About This Tool"):
            st.markdown("""
            This scheduler creates an optimal performance order by:
            
            - **Respecting costume change times**: Ensures dancers have enough time between performances
            - **Keeping siblings on the same day**: Families don't have to attend multiple days
            - **Balancing show length**: Distributes performances evenly across days
            - **Organizing by age**: Creates a natural flow from younger to older performers
            - **Honoring constraints**: Respects same-day and different-day requirements
            
            The optimizer uses integer linear programming (PuLP) to find the best solution.
            """)


if __name__ == "__main__":
    main()
