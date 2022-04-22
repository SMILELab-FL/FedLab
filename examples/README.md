# Quick start with FedLab demos


## Synchronous mode

### Standalone

```
$ cd standalone-mnist
$ bash launch_eg.sh
```
---
### Cross-process

```
$ cd cross-process-mnist
$ bash launch_eg.sh
```

---
### Hierarchical-hybrid-mnist

Run all scripts together:
```
$ cd hierarchical-hybrid-mnist
$ bash launch_eg.sh
```

Run scripts seperately:

Top server (terminal 1):
```
$ bash launch_topserver_eg.sh
```

Scheduler1 + Ordinary trainer with 1 client + Serial trainer with 10 clients (terminal 2):
```
$ bash launch_cgroup1_eg.sh
```

Scheduler2 + Ordinary trainer with 1 client + Serial trainer with 10 clients (terminal 3):
```
$ bash launch_cgroup2_eg.sh
```

## Asynchronous mode

### Cross process

```
$ cd asynchronous-cross-process-mnist
$ bash launch_eg.sh
```