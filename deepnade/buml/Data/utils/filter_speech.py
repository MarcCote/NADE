def is_phone_state_f(phone_state):
    def is_phone_state(acoustics, labels):
        which = labels[:, phone_state] == 1
        return (acoustics[which],)        
    return lambda acoustics, labels: is_phone_state(acoustics, labels)

def is_not_phone_state_f(phone_state):
    def is_not_phone_state(acoustics, labels):
        which = labels[:, phone_state] != 1
        return (acoustics[which],)
    return lambda acoustics, labels: is_not_phone_state(acoustics, labels)