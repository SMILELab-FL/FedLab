CLIENT_HOLDER = 1
SERVER = 1 << 1
BOARD_SHOWER = 1 << 2

ALL = CLIENT_HOLDER | SERVER | BOARD_SHOWER
SERVER_SHOWER = SERVER | BOARD_SHOWER


def is_client_holder(role):
    return bool(role & CLIENT_HOLDER)


def is_server(role):
    return bool(role & SERVER)


def is_board_shower(role):
    return bool(role & BOARD_SHOWER)
