# Return the simulated time it would take for a server to receive all messages
def find_slowest_time(messages):
    return max(map(lambda m: m.simulated_time, messages))
