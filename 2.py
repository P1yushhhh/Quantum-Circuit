from qiskit_ibm_runtime import QiskitRuntimeService
service = QiskitRuntimeService()
print([b.name for b in service.backends()])
