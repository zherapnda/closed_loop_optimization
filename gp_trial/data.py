class ExperimentData:
    def __init__(self, structures):
        self.experiments = []
        self.test_structures = structures


    def add_experiment(self, structure_id, amplitude, rise, dwell, fall, period, dc):
        self.experiments.append({
            'structure_id': structure_id,
            'amplitude': amplitude,
            'rise': rise,
            'dwell': dwell,
            'fall': fall,
            'period': period,
            'dc': dc
        })


    def mark_used(self, structure_id):
        for structure in self.test_structures:
            if structure['id'] == structure_id:
                structure['used'] = True
                break


    def get_available_structures(self):
        return [structure for structure in self.test_structures if not structure.get('used', False)]