import config


def find_slowest_time(messages):
    simulated_communication_times = {message.sender: message.body['simulated_time'] for message in messages}
    slowest_client = max(simulated_communication_times, key=simulated_communication_times.get)
    # simulated time it would take for server to receive all values (server waiting for straggler(s))
    return simulated_communication_times[slowest_client]


def print_config():
    """
    Prints parameters at start of simulation
    """
    main_msg = 'Running FL simulation on {} clients for {} rounds\n'.format(config.NUM_CLIENTS, config.ITERATIONS)
    params_msg = 'Hyperparameters...\nFraction of clients per round = {}\nBatch Size = {}\nLocal Epochs = {}'.format(
        config.CLIENT_FRACTION, config.BATCH_SIZE, config.EPOCHS
    )
    print(main_msg + params_msg + '\n')
