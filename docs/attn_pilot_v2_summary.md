## V2 final-step summary

| variant | params | state | final train loss | final eval loss | final eval ppl | final eval mask_acc | runtime (min) |
|---|---|---|---|---|---|---|---|
| separate-QKV chain-aware | 24,011,264 | finished | 0.4708 | 0.2856 | 1.3306 | 0.9275 | 193.2 |
| shared-QKV chain-aware | 19,292,672 | finished | 0.4735 | 0.2853 | 1.3302 | 0.9265 | 162.8 |
| shared-QKV chain-aware (param-matched) | 23,916,096 | finished | 0.4799 | 0.2865 | 1.3318 | 0.9273 | 170.1 |
| standard MHA (same size) | 19,292,672 | finished | 0.4779 | 0.2840 | 1.3285 | 0.9274 | 140.1 |
| standard MHA (param-matched, d=288) | 23,916,096 | finished | 0.4806 | 0.2844 | 1.3289 | 0.9272 | 147.5 |

### Eval trajectories (every 5k steps)

```
  step         separate_chain_aware           shared_chain_aware        shared_chain_aware_pm               standard_small       standard_param_matched
  5000                       0.3093                       0.3105                       0.3102                       0.3111                       0.3082
 10000                       0.2967                       0.2992                       0.2976                       0.3006                       0.3037
 15000                       0.2946                       0.2931                       0.2939                       0.2939                       0.2941
 20000                       0.2919                       0.2910                       0.2943                       0.2913                       0.2890
 25000                       0.2895                       0.2886                       0.2903                       0.2894                       0.2889
 30000                       0.2884                       0.2892                       0.2893                       0.2868                       0.2884
 35000                       0.2883                       0.2872                       0.2903                       0.2860                       0.2868
 40000                       0.2875                       0.2862                       0.2872                       0.2855                       0.2855
 45000                       0.2857                       0.2855                       0.2861                       0.2855                       0.2855
 50000                       0.2856                       0.2853                       0.2865                       0.2840                       0.2844
```
