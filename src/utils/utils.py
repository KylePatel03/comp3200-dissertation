def find_slowest_time(messages):
    simulated_communication_times = {message.sender: message.body['simulated_time'] for message in messages}
    slowest_client = max(simulated_communication_times, key=simulated_communication_times.get)
    # simulated time it would take for server to receive all values (server waiting for straggler(s))
    return simulated_communication_times[slowest_client]
