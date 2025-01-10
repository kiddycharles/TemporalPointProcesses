import numpy as np
import torch
import torch.utils.data

from transformer import Constants


def compute_time_to_last_same_event(data):
    time_to_last_same_event = []

    last_event_time = {}  # Dictionary to store the last occurrence time for each event type

    for elem in data:
        event_type = elem['type_event']
        time_since_start = elem['time_since_start']

        # Check if the event type has occurred before
        if event_type in last_event_time:
            time_diff = time_since_start - last_event_time[event_type]
        else:
            time_diff = 0  # No previous event of the same type

        # Append the time difference to the result list
        time_to_last_same_event.append(time_diff)

        # Update the last occurrence of the event type
        last_event_time[event_type] = time_since_start

    return time_to_last_same_event


def compute_time_n_gap(data, n):
    """
    Compute the time difference between each event and the n-th previous event in each instance.

    Parameters:
    data (list of lists): Each instance contains a list of events, where each event is a dictionary
                          with 'time_since_start' and 'time_since_last_event' keys.
    n (int): The number of events back to compute the time gap for.

    Returns:
    time_n_gap (list of lists): Each instance contains a list of time differences from the current
                                event to the n-th previous event.
    """
    # Extract times since start
    time_since_start = [[elem['time_since_start'] for elem in inst] for inst in data]

    # Initialize time_n_gap
    time_n_gap = []

    for instance_times in time_since_start:
        # Compute time_n_gap for each instance
        instance_n_gap = []

        for i in range(len(instance_times)):
            # If there are not enough previous events, append None (or 0 if that's preferred)
            if i < n:
                instance_n_gap.append(0.0)  # Or use 0 if you prefer
            else:
                # Calculate time gap with the n-th previous event
                time_gap_n = instance_times[i] - instance_times[i - n]
                instance_n_gap.append(time_gap_n)

        time_n_gap.append(instance_n_gap)

    return time_n_gap


class EventData(torch.utils.data.Dataset):
    """ Event stream dataset. """

    def __init__(self, data):
        """
        Data should be a list of event streams; each event stream is a list of dictionaries;
        each dictionary contains: time_since_start, time_since_last_event, type_event
        """
        self.time = [[elem['time_since_start'] for elem in inst] for inst in data]
        self.time_gap = [[elem['time_since_last_event'] for elem in inst] for inst in data]
        # plus 1 since there could be event type 0, but we use 0 as padding
        self.event_type = [[elem['type_event'] + 1 for elem in inst] for inst in data]
        self.time_to_last_same_event = [compute_time_to_last_same_event(instance) for instance in data]
        self.time_n_gap = compute_time_n_gap(data, 2)

        self.length = len(data)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """ Each returned element is a list, which represents an event stream """
        return self.time[idx], self.time_gap[idx], self.event_type[idx], self.time_to_last_same_event[idx], \
        self.time_n_gap[idx]


def pad_time(insts):
    """ Pad the instance to the max seq length in batch. """

    max_len = max(len(inst) for inst in insts)

    batch_seq = np.array([
        inst + [inst[-1]] * (max_len - len(inst))
        for inst in insts])

    return torch.tensor(batch_seq, dtype=torch.float32)


def pad_type(insts):
    """ Pad the instance to the max seq length in batch. """

    max_len = max(len(inst) for inst in insts)

    batch_seq = np.array([
        inst + [Constants.PAD] * (max_len - len(inst))
        for inst in insts])

    return torch.tensor(batch_seq, dtype=torch.long)


def collate_fn(insts):
    """ Collate function, as required by PyTorch. """

    time, time_gap, event_type, time_to_last, time_n_gap = list(zip(*insts))
    time = pad_time(time)
    time_gap = pad_time(time_gap)
    event_type = pad_type(event_type)
    time_to_last_event_same_type = pad_time(time_to_last)
    time_n_gap = pad_time(time_n_gap)
    # return time, time_gap, event_type, time_to_last_event_same_type, time_n_gap
    return time, time_gap, event_type


def get_dataloader(data, batch_size, shuffle=True):
    """ Prepare dataloader. """

    ds = EventData(data)
    dl = torch.utils.data.DataLoader(
        ds,
        num_workers=2,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle
    )
    return dl

