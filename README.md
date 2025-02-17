# CandyPoker

This is a C++ poker project authored by Gerry Candy, which aims to be a general C++ poker library. This project was started in 2017 with the original goals of,
* Be a fast evaluation library
* Solve three-player push-fold
* Provide an framework for other poker software

The original goals have now been achived

The requirements are boost, and Eigen. Although Eigen is just used for Vector/Matrix multiplication.

# Getting Started Windows
        
        REM you first need to clone https://gitlab.com/libeigen/eigen.git to somewhere
        REM also get some version of boost 
        git clone https://github.com/sweeterthancandy/CandyPoker.git
        cd CandyPoker
        mkdir deps
        cd deps
        git clone https://gitlab.com/libeigen/eigen.git
        git clone https://github.com/google/googletest.git
        git clone https://github.com/google/benchmark.git
        cd ..
        cmake -B build -DCMAKE_BUILD_TYPE=Release -DBoost_INCLUDE_DIR=C:\work\boost_1_77_0\boost_1_77_0
        REM open build\ps.sln and build
        

# Getting Started Linux


        git@github.com:sweeterthancandy/CandyPoker.git
        cd CandyPoker
        mkdir deps
        cd deps
        git clone https://github.com/google/googletest.git
        cd ..
        cmake -G Ninja -DCMAKE_BUILD_TYPE=Release .
        ninja



        # simple equity evaluation
        ./candy-poker eval AA KK 
        |range| equity |  wins  |draw_1|draw equity| sigma  |
        +-----+--------+--------+------+-----------+--------+
        | AA  |81.9461%|50371344|285228|142614.00% |61642944|
        | KK  |18.0539%|10986372|285228|142614.00% |61642944|
        # create binary all in equity cache
        ./candy-poker read-cache --file Assets/PushFoldEquity.json

        # now create the cache
        ./candy-poker read-cache
        mv .cc.bin.stage .cc.bin

        # now create HU push-fold table
        ./candy-poker solver --solver quick-solver --eff-lower 1.0 --eff-upper 20.0 --eff-inc 1.0 --cum-table --print-seq

                               sb push/fold

            A    K    Q    J    T    9    8    7    6    5    4    3    2   
          +-----------------------------------------------------------------
        A |20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0
        K |20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 19.0 19.0
        Q |20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 16.0 13.0 12.0
        J |20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 18.0 17.0 13.0 10.0 8.0 
        T |20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 11.0 10.0 7.0  6.0 
        9 |20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 14.0 5.0  4.0  3.0 
        8 |20.0 18.0 13.0 13.0 16.0 20.0 20.0 20.0 20.0 18.0 8.0  2.0  2.0 
        7 |20.0 16.0 10.0 8.0  9.0  10.0 14.0 20.0 20.0 20.0 14.0 2.0  2.0 
        6 |20.0 15.0 9.0  6.0  5.0  4.0  7.0  10.0 20.0 20.0 16.0 7.0  2.0 
        5 |20.0 14.0 8.0  6.0  4.0  3.0  2.0  2.0  2.0  20.0 20.0 12.0 2.0 
        4 |20.0 13.0 8.0  5.0  3.0  2.0  2.0  2.0  2.0  2.0  20.0 9.0   1  
        3 |20.0 12.0 7.0  5.0  3.0  2.0   1    1    1    1    1   20.0  1  
        2 |20.0 11.0 7.0  4.0  2.0  2.0   1    1    1    1    1    1   20.0

                          bb call/fold, given bb push

            A    K    Q    J    T    9    8    7    6    5    4    3    2   
          +-----------------------------------------------------------------
        A |20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0
        K |20.0 20.0 20.0 20.0 20.0 20.0 17.0 15.0 14.0 13.0 12.0 11.0 10.0
        Q |20.0 20.0 20.0 20.0 20.0 16.0 13.0 10.0 10.0 8.0  8.0  7.0  7.0 
        J |20.0 20.0 19.0 20.0 18.0 13.0 10.0 8.0  7.0  6.0  6.0  5.0  5.0 
        T |20.0 20.0 14.0 12.0 20.0 11.0 9.0  7.0  6.0  5.0  5.0  4.0  4.0 
        9 |20.0 17.0 11.0 9.0  8.0  20.0 8.0  6.0  5.0  5.0  4.0  4.0  3.0 
        8 |20.0 13.0 9.0  7.0  6.0  6.0  20.0 6.0  5.0  4.0  4.0  3.0  3.0 
        7 |20.0 12.0 7.0  6.0  5.0  4.0  4.0  20.0 5.0  4.0  4.0  3.0  3.0 
        6 |20.0 11.0 7.0  5.0  4.0  4.0  4.0  4.0  20.0 4.0  4.0  3.0  3.0 
        5 |20.0 10.0 6.0  5.0  4.0  3.0  3.0  3.0  3.0  20.0 4.0  4.0  3.0 
        4 |18.0 9.0  6.0  4.0  3.0  3.0  3.0  3.0  3.0  3.0  20.0 3.0  3.0 
        3 |16.0 8.0  5.0  4.0  3.0  3.0  2.0  2.0  2.0  3.0  3.0  20.0 3.0 
        2 |15.0 8.0  5.0  4.0  3.0  3.0  2.0  2.0  2.0  2.0  2.0  2.0  15.0



## Poker Evaluation

The poker evaulation doesn't support any boards, as this is mostly noise in the code as the preflop all-in with 5 cards to be dealt is the target case.

        ./candy-poker eval TT+,AQs+,AQo+ QQ+,AKs,AKo TT
        |    range    | equity |   wins   | draw_1  | draw_2 | draw equity |   sigma   |
        +-------------+--------+----------+---------+--------+-------------+-----------+
        |TT+,AQs+,AQo+|30.8450%|3213892992|566688312|41308668|297113712.00%|11382741216|
        | QQ+,AKs,AKo |43.0761%|4637673516|503604216|41308668|265571664.00%|11382741216|
        |     TT      |26.0789%|2923177728|63084096 |41308668|45311604.00% |11382741216|
         3.611110s wall, 3.580000s user + 0.010000s system = 3.590000s CPU (99.4%)


## Two Player Push Fold EV


For solving push/fold games, a distinction has to be made between mixed solutions and minimally-mixed solutions. From a game theory perspective there should exist a GTO solution where each decision has only one holdem hand type mixed, meaning we can partition a strategy into {PUSH,FOLD,MIXED}, where MIXED has only one holdem hand type like 68s etc. What this means is that, it we are only looking for any solution, we can run a algorithm of

        Stategry FindAnyGTO(Stategry S){
                double factor = 0.05;
                for(; SomeCondition(S); ){
                        auto counter = CounterStrategy(S);
                        S = S * ( 1- factor ) + counter * factor;
                }
                return S;
        }

The above will converge to a GTO solution, if you where to program a computer to play poker, however we want to find a minimally mixed solution. This is a much hard problem, as there is no simple algorithm which would converge to a minially mixed solution because for any solution, there are going to be a subset of hands which are almost indifferent to being wither PUSH or FOLD, causing computation problems.

Another consideration when looking for solutions, is that although most solvers are compoutaitionally tangible for two player, three player push fold is much much slower, we have to create a suitable algorith. 

The solution to these algorithms, was to chain several different algorithms together, with special metrics. For this I developed these concepts

### Solution.Gamma

The Gamma for a solution is the number of hands, for which the counterstrategy is different. From the game theory we know that their exists a GTO mixed solution, and also we know that there exists a GTO solution with each player only having one mixed card (I think thats right). Gamma is an array of sets corresponding to each player. For example for HU, we might have a gamma of {{Q5s}, {64o,QJo}}, which would indicate that for the SB the hand Q5s is pretty indifferent to PUSH/FOLD, and also for BB 64o and QJo is indifferent to PUSH/FOLD.

### Solution.Level, Solution.Total

The Level of a solution is the maximum number of Gamma cards for each player. So a gamma vector of {{Q5s}, {64o,QJo}} would correspond to (1,2), so the level of the solution is 2, and the total is 3.

### Solution Sequence

Find GTO poker solutions is a stochastic process, as we are finding the find the best solution possible. This means that we are basically producing a sequence of candidate solutions {A0,A1,A2,...}, and from this we create a best to date sequence {B0,B1,B2,...}. 

We have three solvers
* simple-numeric
* single-permutation
* permutation

#### simple-numeric

This is the most basic solver, and is the basic FindAnyGTO() function that has been discussed. As an implementation detail we create an infinite loop, keeping taking the linear product of the two solutions, with a time to live variable of say 10, then if we have 10 iterations with a new best-to-date solution, we return that solution. 


### single-permutation

This is an algebraic solver, each takes the trail solution S, and creates a set with each gamma cards (a particilar card for with S is different from CounterStrategy(S)), with the trail solutions card replaced with either PUSH or FOLD, with all other cards the same. This means that with a gamma vector of {{Q5s}, {64o,QJo}} we would have 2^3 candidate solutions { S with Q5s for player 1 FOLD,  S with Q5s for player 1 PUSH, S with 64o for player 2 PUSH, ...}. We then see if any of these new strategies is a "better" solution, if this is the case we restart the algorith. This algorithm would find the specific solution, depending on the path taken by the best-to-date sequence. 

### permutation

This is another algebraic solver, which tries to replace take the Gamma vector, and replace all but 1 card per decision with a mixed solution, and discretized the mixed solution with a grid. 

For eaxmple with a Gamma vector of {{Q5s}, {64o,QJo}}, we would have gamma vectors of 

                (m)(ff)
                (m)(pf)
                (m)(fp)
                (m)(pp)
                (f)(mf)
                (p)(mf)
                (f)(mp)
                (p)(mp)
                (f)(fm)
                (p)(fm)
                (f)(pm)
                (p)(pm),

where in the above, we replace each m with one of (0,1/n,2/m,..,(n-1)/n,n). We then take the best solution

### Aggregate solvers

After much expreimentation I found that the best thing was a mix of different solvers, for which are described in a JSON file called .ps.solvers.json.
Below is the description for the accurate-solver. Here we first apply the numeric solver with a factor of 0.4 (which means take 40% of the new solution),
by considering a total-sequence best-to-date. With ttl ~ time-to-live of 1, and a stride of 20, this means that is 


        "accurate-solver":[
                { "solver":"numeric-sequence", "args": { "clamp-epsilon":1e-3, "sequence-type":"total-sequence", "factor":0.4 , "ttl":1, "stride":20  } },
                { "solver":"numeric-sequence", "args": { "clamp-epsilon":1e-3, "sequence-type":"total-sequence", "factor":0.2 , "ttl":1, "stride":40  } },
                { "solver":"numeric-sequence", "args": { "clamp-epsilon":1e-3, "sequence-type":"total-sequence", "factor":0.1 , "ttl":1, "stride":80  } },
                { "solver":"single-permutation" },
                { "solver":"permutation", "args":{ "grid-size":1 } }
        ],


        for(size_t loop_count{1};;++loop_count){
                for(size_t inner=0;inner!=args_.stride;++inner){
                        auto S_counter = computation_kernel::CounterStrategy(gt, AG, S, args_.delta);
                        computation_kernel::InplaceLinearCombination(S, S_counter, 1 - args_.factor );
                }
                computation_kernel::InplaceClamp(S, args_.clamp_epsilon);
                        
                auto solution = Solution::MakeWithDeps(gt, AG, S);
                for(auto& ctrl : controllers_ ){
                        auto opt_opt = ctrl->Apply(loop_count, gt, AG, args_, solution);
                        if( opt_opt ){
                                return *opt_opt;
                        }
                }

        }



For a quick example, we can run the command

        ./driver scratch --game-tree two-player-push-fold --solver pipeline --eff-lower 2 --eff-upper 20 --eff-inc 1 --cum-table

                    sb push/fold

            A    K    Q    J    T    9    8    7    6    5    4    3    2
          +-----------------------------------------------------------------
        A |20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0
        K |20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 19.8 19.3
        Q |20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 19.9 16.3 13.5 12.7
        J |20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 18.6 15.3 13.5 10.6 8.5
        T |20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 11.9 10.6 7.5  6.5
        9 |20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 14.4 6.5  4.9  3.7
        8 |20.0 18.0 13.0 12.9 17.4 20.0 20.0 20.0 20.0 18.8 10.0 2.7  2.5
        7 |20.0 16.1 10.3 8.5  9.0  10.8 15.2 20.0 20.0 20.0 13.9 2.5  2.1
        6 |20.0 15.0 9.6  6.5  5.7  5.2  7.0  10.7 20.0 20.0 16.3 7.1  2.0
        5 |20.0 14.2 8.9  6.0  4.1  3.4  3.0  2.6  2.4  20.0 20.0 13.3 2.0
        4 |20.0 13.1 7.9  5.4  3.8  2.7  2.3  2.1  2.0  2.1  20.0 9.9   0
        3 |20.0 12.1 7.5  5.0  3.4  2.5   0    0    0    0    0   20.0  0
        2 |20.0 11.6 7.0  4.6  2.9  2.2   0    0    0    0    0    0   20.0

                    bb call/fold, given bb push

            A    K    Q    J    T    9    8    7    6    5    4    3    2
          +-----------------------------------------------------------------
        A |20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0
        K |20.0 20.0 20.0 20.0 20.0 20.0 17.6 15.3 14.3 13.2 12.0 11.4 10.7
        Q |20.0 20.0 20.0 20.0 20.0 16.0 13.0 10.5 9.9  8.9  8.4  7.8  7.2
        J |20.0 20.0 19.1 20.0 18.1 13.3 10.6 8.8  7.0  6.9  6.1  5.8  5.5
        T |20.0 20.0 14.8 12.6 20.0 11.5 9.2  7.4  6.3  5.2  5.1  4.8  4.5
        9 |20.0 16.9 11.6 9.5  8.3  20.0 8.3  6.9  5.8  5.0  4.3  4.1  3.9
        8 |20.0 13.7 9.7  7.6  6.6  6.0  20.0 6.5  5.6  4.8  4.1  3.6  3.5
        7 |20.0 12.2 7.9  6.3  5.4  4.9  4.7  20.0 5.4  4.8  4.1  3.6  3.3
        6 |20.0 10.9 7.3  5.3  4.6  4.2  4.1  4.0  20.0 4.9  4.3  3.8  3.3
        5 |20.0 10.2 6.8  5.1  4.0  3.7  3.6  3.6  3.7  20.0 4.6  4.0  3.6
        4 |18.2 9.1  6.2  4.7  3.8  3.3  3.2  3.2  3.3  3.5  20.0 3.8  3.4
        3 |16.4 8.6  5.8  4.4  3.6  3.1  2.9  2.9  2.9  3.1  3.0  20.0 3.3
        2 |15.6 8.1  5.6  4.2  3.5  3.0  2.8  2.6  2.7  2.8  2.7  2.6  15.0


What is actually much more of a problem is finding a solution with the least number of mixed solutions. Taking BB of 10 for the example, it's not too difficul to implement a solver which finds a game-theoritic solution to the push fold solution
        
        ./driver scratch --solver D --eff-lower 1.0 --eff-upper 20.0 --cum-table --print-seq


#### --print-seq

The print-seq option shows a table with metrics for each solution. Below we can see that each solution only has one card different in the Gamma vector.

        |           Desc           | ? |Level|Total|Gamma |  GammaCards  |Mixed |MixedCards |         |.|          |
        +--------------------------+---+-----+-----+------+--------------+------+-----------+----------------------+
        |GameTreeTwoPlayer:0.5:1:1 |yes|  0  |  0  |{0, 0}|   {{}, {}}   |{0, 0}| {{}, {}}  |          0           |
        |GameTreeTwoPlayer:0.5:1:2 |yes|  0  |  0  |{0, 0}|   {{}, {}}   |{0, 0}| {{}, {}}  |          0           |
        |GameTreeTwoPlayer:0.5:1:3 |yes|  1  |  1  |{0, 1}| {{}, {63o}}  |{0, 1}|{{}, {63o}}|2.7882113404584241e-05|
        |GameTreeTwoPlayer:0.5:1:4 |yes|  0  |  0  |{0, 0}|   {{}, {}}   |{0, 0}| {{}, {}}  |          0           |
        |GameTreeTwoPlayer:0.5:1:5 |yes|  1  |  1  |{0, 1}| {{}, {97o}}  |{0, 1}|{{}, {97o}}|1.8804630453818361e-05|
        |GameTreeTwoPlayer:0.5:1:6 |yes|  1  |  1  |{0, 1}| {{}, {98o}}  |{0, 1}|{{}, {98o}}|1.7960296257168995e-08|
        |GameTreeTwoPlayer:0.5:1:7 |yes|  1  |  1  |{0, 1}| {{}, {97s}}  |{0, 1}|{{}, {97s}}|1.1004535174736346e-05|
        |GameTreeTwoPlayer:0.5:1:8 |yes|  1  |  1  |{0, 1}| {{}, {Q7o}}  |{0, 1}|{{}, {Q7o}}|0.00025938863366876778|
        |GameTreeTwoPlayer:0.5:1:9 |yes|  0  |  0  |{0, 0}|   {{}, {}}   |{0, 0}| {{}, {}}  |          0           |
        |GameTreeTwoPlayer:0.5:1:10|yes|  1  |  1  |{0, 1}| {{}, {Q6s}}  |{0, 0}| {{}, {}}  |5.8276293633093001e-05|
        |GameTreeTwoPlayer:0.5:1:11|yes|  1  |  1  |{0, 1}| {{}, {K6o}}  |{0, 0}| {{}, {}}  |1.8685953474206762e-05|
        |GameTreeTwoPlayer:0.5:1:12|yes|  0  |  0  |{0, 0}|   {{}, {}}   |{0, 0}| {{}, {}}  |          0           |
        |GameTreeTwoPlayer:0.5:1:13|yes|  1  |  1  |{0, 1}| {{}, {JTo}}  |{0, 0}| {{}, {}}  |8.3942868595962561e-05|
        |GameTreeTwoPlayer:0.5:1:14|yes|  1  |  1  |{1, 0}| {{J8o}, {}}  |{0, 0}| {{}, {}}  |9.9252283580320011e-05|
        |GameTreeTwoPlayer:0.5:1:15|yes|  1  |  1  |{0, 1}| {{}, {QTo}}  |{0, 1}|{{}, {QTo}}|6.5683741556160635e-05|
        |GameTreeTwoPlayer:0.5:1:16|yes|  1  |  1  |{0, 1}| {{}, {A2o}}  |{0, 0}| {{}, {}}  |0.00010373800344154471|
        |GameTreeTwoPlayer:0.5:1:17|yes|  1  |  2  |{1, 1}|{{J5s}, {K9o}}|{0, 0}| {{}, {}}  |0.00018949470911927557|
        |GameTreeTwoPlayer:0.5:1:18|yes|  1  |  1  |{1, 0}| {{J6s}, {}}  |{0, 0}| {{}, {}}  |1.2054322645116411e-05|
        |GameTreeTwoPlayer:0.5:1:19|yes|  0  |  0  |{0, 0}|   {{}, {}}   |{0, 0}| {{}, {}}  |          0           |
        |GameTreeTwoPlayer:0.5:1:20|yes|  1  |  1  |{0, 1}| {{}, {QJo}}  |{0, 0}| {{}, {}}  |2.1377144644924018e-05|

### Three player Pre-flop all in-EV

This was the original goal of the project. First consider the game tree in the below table. We have 7 terminal states.

        |   Path   | Impl | Active  |    Pot    |
        +----------+------+---------+-----------+
        |  *,f,ff  |Static|   {2}   | [0,0.5,1] |
        |*,f,fp,fpf|Static|   {1}   | [0,10,1]  |
        |*,f,fp,fpp| Eval | {1, 2}  | [0,10,10] |
        |*,p,pf,pff|Static|   {0}   |[10,0.5,1] |
        |*,p,pf,pfp| Eval | {0, 2}  |[10,0.5,10]|
        |*,p,pp,ppf| Eval | {0, 1}  | [10,10,1] |
        |*,p,pp,ppp| Eval |{0, 1, 2}|[10,10,10] |

This tree is represented by a 6-vector

        (BTN P/F, SB P/F Given BTN P, SB P/F Given BTN F, SB C Given BTN P, SB P, SB C Given BTN P, SB F, SB C Given BTN F, SB P)


## FlopZilla
        ./candy-poker flopzilla ATs+,AJo+,88+,QJs,KQo
        +-------------------+-------+----------------------+----------------------+
        |       Rank        | Count |         Prob         |         Cum          |
        +-------------------+-------+----------------------+----------------------+
        |    Royal Flush    |  20   |0.00092764378478664194|0.00092764378478664194|
        |  Straight Flush   |   8   |0.00037105751391465676|0.0012987012987012987 |
        |       Quads       | 2152  | 0.099814471243042671 | 0.10111317254174397  |
        |    Full House     | 9288  |  0.4307977736549165  | 0.53191094619666046  |
        |       Flush       | 3272  | 0.15176252319109462  | 0.68367346938775508  |
        |     Straight      | 5604  | 0.25992578849721709  | 0.94359925788497212  |
        |       Trips       |109648 |  5.0857142857142854  |  6.0293135435992582  |
        |     Two pair      |186912 |   8.66938775510204   |  14.698701298701298  |
        |     One pair      |1129920|  52.408163265306122  |  67.106864564007424  |
        |     High Card     |709176 |  32.893135435992576  |         100          |
        +-------------------+-------+----------------------+----------------------+
        |    Flush Draw     | Count |         Prob         |         Cum          |
        +-------------------+-------+----------------------+----------------------+
        |       Flush       | 3300  | 0.15306122448979592  | 0.15306122448979592  |
        |    Four-Flush     | 82500 |  3.8265306122448979  |  3.9795918367346941  |
        |    Three-Flush    |683100 |  31.683673469387756  |  35.663265306122447  |
        |       None        |1387100|  64.336734693877546  |         100          |
        +-------------------+-------+----------------------+----------------------+
        |   Straight Draw   | Count |         Prob         |         Cum          |
        +-------------------+-------+----------------------+----------------------+
        |     Straight      | 5632  | 0.26122448979591839  | 0.26122448979591839  |
        |   Four-Straight   | 66112 |  3.0664192949907236  |  3.327643784786642   |
        |   Double-Gutter   | 2560  | 0.11873840445269017  |  3.446382189239332   |
        |       None        |2081696|  96.55361781076067   |         100          |
        +-------------------+-------+----------------------+----------------------+
        |   Detailed Rank   | Count |         Prob         |         Cum          |
        +-------------------+-------+----------------------+----------------------+
        |    Royal Flush    |  20   |0.00092764378478664194|0.00092764378478664194|
        |  Straight Flush   |   8   |0.00037105751391465676|0.0012987012987012987 |
        |       Quads       | 2152  | 0.099814471243042671 | 0.10111317254174397  |
        |    Full House     | 9288  |  0.4307977736549165  | 0.53191094619666046  |
        |       Flush       | 3272  | 0.15176252319109462  | 0.68367346938775508  |
        |     Straight      | 5604  | 0.25992578849721709  | 0.94359925788497212  |
        |       Trips       |109648 |  5.0857142857142854  |  6.0293135435992582  |
        |     Two pair      |186912 |   8.66938775510204   |  14.698701298701298  |
        |     One pair      |1129920|  52.408163265306122  |  67.106864564007424  |
        |High Card - 2 Overs|501780 |  23.273654916512058  |  90.380519480519482  |
        |High Card - 1 Over |153696 |  7.1287569573283855  |  97.509276437847873  |
        |High Card - Unders | 53700 |  2.4907235621521338  |         100          |
        +-------------------+-------+----------------------+----------------------+


        ./candy-poker flopzilla 100%
        +-------------------+--------+----------------------+----------------------+
        |       Rank        | Count  |         Prob         |         Cum          |
        +-------------------+--------+----------------------+----------------------+
        |    Royal Flush    |   40   |0.00015390771693292701|0.00015390771693292701|
        |  Straight Flush   |  360   |0.0013851694523963432 |0.0015390771693292702 |
        |       Quads       |  6240  | 0.024009603841536616 | 0.025548681010865885 |
        |    Full House     | 37440  | 0.14405762304921968  | 0.16960630406008556  |
        |       Flush       | 51080  |  0.1965401545233478  | 0.36614645858343337  |
        |     Straight      | 102000 | 0.39246467817896391  | 0.75861113676239722  |
        |       Trips       | 549120 |  2.112845138055222   |  2.8714562748176196  |
        |     Two pair      |1235520 |  4.7539015606242501  |  7.6253578354418687  |
        |     One pair      |10982400|  42.256902761104442  |  49.88226059654631   |
        |     High Card     |13025400|  50.11773940345369   |         100          |
        +-------------------+--------+----------------------+----------------------+
        |    Flush Draw     | Count  |         Prob         |         Cum          |
        +-------------------+--------+----------------------+----------------------+
        |       Flush       | 51480  | 0.19807923169267708  | 0.19807923169267708  |
        |    Four-Flush     |1115400 |  4.2917166866746701  |  4.4897959183673466  |
        |    Three-Flush    |8477040 |  32.617046818727488  |  37.106842737094837  |
        |       None        |16345680|  62.893157262905163  |         100          |
        +-------------------+--------+----------------------+----------------------+
        |   Straight Draw   | Count  |         Prob         |         Cum          |
        +-------------------+--------+----------------------+----------------------+
        |     Straight      | 92160  | 0.35460337981346385  | 0.35460337981346385  |
        |   Four-Straight   | 890880 |  3.4278326715301506  |  3.7824360513436144  |
        |   Double-Gutter   | 71680  | 0.27580262874380523  |  4.0582386800874195  |
        |       None        |24934880|  95.941761319912587  |         100          |
        +-------------------+--------+----------------------+----------------------+
        |   Detailed Rank   | Count  |         Prob         |         Cum          |
        +-------------------+--------+----------------------+----------------------+
        |    Royal Flush    |   40   |0.00015390771693292701|0.00015390771693292701|
        |  Straight Flush   |  360   |0.0013851694523963432 |0.0015390771693292702 |
        |       Quads       |  6240  | 0.024009603841536616 | 0.025548681010865885 |
        |    Full House     | 37440  | 0.14405762304921968  | 0.16960630406008556  |
        |       Flush       | 51080  |  0.1965401545233478  | 0.36614645858343337  |
        |     Straight      | 102000 | 0.39246467817896391  | 0.75861113676239722  |
        |       Trips       | 549120 |  2.112845138055222   |  2.8714562748176196  |
        |     Two pair      |1235520 |  4.7539015606242501  |  7.6253578354418687  |
        |     One pair      |10982400|  42.256902761104442  |  49.88226059654631   |
        |High Card - 2 Overs|1302540 |  5.0117739403453685  |  54.894034536891681  |
        |High Card - 1 Over |3907620 |  15.035321821036106  |  69.929356357927787  |
        |High Card - Unders |7815240 |  30.070643642072213  |         100          |
        +-------------------+--------+----------------------+----------------------+


# Some More soltuions

## Two player non-mixed 




                    sb push/fold

             A     K     Q     J     T     9    8    7    6     5    4    3    2   
          +------------------------------------------------------------------------
        A |905.9 622.8 318.4 147.1 833.9 80.0  71.9 59.8 53.4 141.8 72.0 57.1 50.5
        K |469.3 833.2 145.7 114.3 92.4  75.2  41.4 39.7 36.2 30.5  25.1 19.9 18.6
        Q |137.3 128.1 318.4 92.4  84.9  48.4  39.9 29.7 29.4 20.0  14.9 13.5 12.5
        J |833.2 78.3  44.8  191.9 87.4  71.5  43.2 31.4 17.0 14.7  12.9 10.7 8.5 
        T |71.2  39.1  38.9  40.5  137.3 72.0  43.8 35.3 23.0 11.9  10.7 7.5  6.5 
        9 |71.2  22.3  20.2  25.2  30.9  112.3 43.9 36.1 25.8 14.4  6.5  4.9  3.7 
        8 |39.7  17.1  13.2  12.0  15.1  20.2  90.6 39.6 29.8 17.7  9.6  2.7  2.5 
        7 |36.9  17.7  10.2   8.5   9.0  10.6  14.7 78.0 35.7 21.8  13.8 2.5  2.1 
        6 |32.5  15.8   9.8   6.5   5.7   5.2  7.0  10.7 90.7 28.0  15.7 7.1  2.0 
        5 |36.9  14.2   8.9   6.0   4.1   3.5  2.9  2.6  2.4  75.4  21.7 12.3 2.0 
        4 |32.5  13.2   8.0   5.3   3.8   2.7  2.3  2.1  2.0   2.1  73.3 9.6  1.8 
        3 |32.3  12.4   7.5   5.1   3.4   2.5  1.9  1.8  1.7   1.8  1.6  71.2 1.7 
        2 |29.3  11.7   7.0   4.6   2.9   2.2  1.8  1.6  1.5   1.5  1.4  1.4  61.1

                    bb call/fold, given bb push

             A     K     Q     J     T     9    8    7    6    5    4    3    2   
          +-----------------------------------------------------------------------
        A |905.9 833.2 128.1 78.0  51.8  40.9  38.4 33.4 31.2 30.2 25.6 23.0 22.2
        K |833.2 833.2 42.5  38.4  32.1  21.1  17.5 15.2 14.3 13.1 12.1 11.5 10.8
        Q |112.3 59.8  331.9 26.2  20.6  15.1  12.3 10.5 10.0 8.9  8.3  7.8  7.2 
        J |67.4  29.3  18.9  195.2 17.3  12.6  10.6 8.6  7.0  6.7  6.2  5.7  5.5 
        T |47.9  24.7  16.4  13.5  138.9 11.5  9.3  7.3  6.4  5.2  5.1  4.8  4.5 
        9 |71.2  20.0  11.8   9.5   8.5  117.0 8.3  6.9  5.8  5.0  4.3  4.1  3.9 
        8 |35.1  14.0   9.9   7.6   6.9   6.0  90.5 6.6  5.6  4.8  4.1  3.6  3.5 
        7 |27.9  12.8   7.9   6.3   5.6   4.9  4.7  62.6 5.5  4.8  4.1  3.6  3.3 
        6 |21.6  11.1   7.4   5.4   4.6   4.2  4.1  4.0  54.7 4.9  4.2  3.8  3.3 
        5 |23.6  10.3   6.7   5.1   4.0   3.7  3.6  3.6  3.7  39.0 4.6  4.0  3.6 
        4 |18.2   9.2   6.2   4.7   3.8   3.3  3.2  3.2  3.3  3.5  32.3 3.8  3.4 
        3 |16.8   8.8   5.9   4.5   3.6   3.1  2.9  2.9  2.9  3.1  3.0  22.3 3.3 
        2 |15.6   8.2   5.5   4.2   3.5   3.0  2.8  2.6  2.7  2.8  2.7  2.6  15.1



## Three player non-mixed



        |             Desc             | ? |Level|Total|      Gamma       |                                            GammaCards                                             |      Mixed       |                  MixedCards                  |         |.|          |
        +------------------------------+---+-----+-----+------------------+---------------------------------------------------------------------------------------------------+------------------+----------------------------------------------+----------------------+
        | GameTreeThreePlayer:0.5:1:1  |yes|  0  |  0  |{0, 0, 0, 0, 0, 0}|                                     {{}, {}, {}, {}, {}, {}}                                      |{0, 0, 0, 0, 0, 0}|           {{}, {}, {}, {}, {}, {}}           |          0           |
        |GameTreeThreePlayer:0.5:1:1.5 |yes|  0  |  0  |{0, 0, 0, 0, 0, 0}|                                     {{}, {}, {}, {}, {}, {}}                                      |{0, 0, 0, 0, 0, 0}|           {{}, {}, {}, {}, {}, {}}           |          0           |
        | GameTreeThreePlayer:0.5:1:2  |yes|  0  |  0  |{0, 0, 0, 0, 0, 0}|                                     {{}, {}, {}, {}, {}, {}}                                      |{0, 0, 0, 0, 0, 0}|           {{}, {}, {}, {}, {}, {}}           |          0           |
        |GameTreeThreePlayer:0.5:1:2.5 |yes|  0  |  0  |{0, 0, 0, 0, 0, 0}|                                     {{}, {}, {}, {}, {}, {}}                                      |{0, 0, 0, 0, 0, 0}|           {{}, {}, {}, {}, {}, {}}           |          0           |
        | GameTreeThreePlayer:0.5:1:3  |yes|  0  |  0  |{0, 0, 0, 0, 0, 0}|                                     {{}, {}, {}, {}, {}, {}}                                      |{0, 0, 0, 0, 0, 0}|           {{}, {}, {}, {}, {}, {}}           |          0           |
        |GameTreeThreePlayer:0.5:1:3.5 |yes|  1  |  2  |{0, 0, 0, 0, 1, 1}|                                  {{}, {}, {}, {}, {J4o}, {Q6o}}                                   |{0, 0, 0, 0, 1, 1}|        {{}, {}, {}, {}, {J4o}, {Q6o}}        |6.6068132444155325e-06|
        | GameTreeThreePlayer:0.5:1:4  |yes|  0  |  0  |{0, 0, 0, 0, 0, 0}|                                     {{}, {}, {}, {}, {}, {}}                                      |{0, 0, 0, 0, 0, 0}|           {{}, {}, {}, {}, {}, {}}           |          0           |
        |GameTreeThreePlayer:0.5:1:4.5 |yes|  1  |  2  |{1, 0, 0, 1, 0, 0}|                                  {{Q8o}, {}, {}, {Q9o}, {}, {}}                                   |{1, 0, 0, 1, 0, 0}|        {{Q8o}, {}, {}, {Q9o}, {}, {}}        |0.00034416030591315899|
        | GameTreeThreePlayer:0.5:1:5  |yes|  1  |  2  |{0, 1, 1, 0, 0, 0}|                                  {{}, {96o}, {J5o}, {}, {}, {}}                                   |{0, 0, 1, 0, 0, 0}|         {{}, {}, {J5o}, {}, {}, {}}          |3.5780305066679396e-05|
        |GameTreeThreePlayer:0.5:1:5.5 |yes|  2  |  6  |{0, 2, 1, 1, 2, 0}|                          {{}, {96o, 86o}, {J2s}, {JTo}, {K5o, J9o}, {}}                           |{0, 2, 1, 1, 2, 0}|{{}, {96o, 86o}, {J2s}, {JTo}, {K5o, J9o}, {}}|0.00029216986864244454|
        | GameTreeThreePlayer:0.5:1:6  |yes|  1  |  4  |{0, 1, 1, 1, 1, 0}|                               {{}, {86o}, {J7o}, {J9s}, {Q6s}, {}}                                |{0, 0, 1, 1, 1, 0}|      {{}, {}, {J7o}, {J9s}, {Q6s}, {}}       |4.5135726283840327e-05|
        |GameTreeThreePlayer:0.5:1:6.5 |yes|  5  | 17  |{3, 3, 0, 5, 2, 4}|{{K7o, Q9o, J7s}, {T2s, 94s, 63s}, {}, {A2o, K9o, K7s, QTo, J9s}, {K2s, Q7s}, {A7o, J7s, 75s, 54s}}|{0, 0, 0, 1, 0, 0}|         {{}, {}, {}, {J9s}, {}, {}}          |0.0028863992003945182 |
        | GameTreeThreePlayer:0.5:1:7  |yes|  2  |  2  |{2, 0, 0, 0, 0, 0}|                                 {{K8o, T9o}, {}, {}, {}, {}, {}}                                  |{2, 0, 0, 0, 0, 0}|       {{K8o, T9o}, {}, {}, {}, {}, {}}       |0.0021832082854016943 |
        |GameTreeThreePlayer:0.5:1:7.5 |yes|  1  |  3  |{1, 0, 1, 1, 0, 0}|                                 {{Q6s}, {}, {J8o}, {QJo}, {}, {}}                                 |{0, 0, 1, 0, 0, 0}|         {{}, {}, {J8o}, {}, {}, {}}          |0.0006248547519683012 |
        | GameTreeThreePlayer:0.5:1:8  |yes|  1  |  1  |{0, 0, 1, 0, 0, 0}|                                    {{}, {}, {K2o}, {}, {}, {}}                                    |{0, 0, 1, 0, 0, 0}|         {{}, {}, {K2o}, {}, {}, {}}          |3.8645665093733905e-05|
        |GameTreeThreePlayer:0.5:1:8.5 |yes|  2  |  5  |{1, 1, 1, 2, 0, 0}|                             {{K4s}, {J7o}, {98s}, {A6o, A5o}, {}, {}}                             |{0, 0, 0, 0, 0, 0}|           {{}, {}, {}, {}, {}, {}}           |0.00029470556607485987|
        | GameTreeThreePlayer:0.5:1:9  |yes|  1  |  2  |{1, 1, 0, 0, 0, 0}|                                  {{K9o}, {43s}, {}, {}, {}, {}}                                   |{0, 0, 0, 0, 0, 0}|           {{}, {}, {}, {}, {}, {}}           |0.0019125361667366308 |
        |GameTreeThreePlayer:0.5:1:9.5 |yes|  1  |  2  |{1, 1, 0, 0, 0, 0}|                                  {{K9o}, {Q6o}, {}, {}, {}, {}}                                   |{0, 0, 0, 0, 0, 0}|           {{}, {}, {}, {}, {}, {}}           |0.0018505557029703523 |
        | GameTreeThreePlayer:0.5:1:10 |yes|  1  |  1  |{0, 0, 1, 0, 0, 0}|                                    {{}, {}, {K5o}, {}, {}, {}}                                    |{0, 0, 1, 0, 0, 0}|         {{}, {}, {K5o}, {}, {}, {}}          |0.0001202869803124812 |
        |GameTreeThreePlayer:0.5:1:10.5|yes|  2  |  5  |{0, 2, 2, 0, 1, 0}|                            {{}, {97o, 76o}, {Q7s, J8s}, {}, {JTs}, {}}                            |{0, 0, 0, 0, 0, 0}|           {{}, {}, {}, {}, {}, {}}           |9.9034768062306044e-05|
        | GameTreeThreePlayer:0.5:1:11 |yes|  1  |  2  |{0, 0, 0, 1, 1, 0}|                                  {{}, {}, {}, {KJo}, {KTo}, {}}                                   |{0, 0, 0, 1, 1, 0}|        {{}, {}, {}, {KJo}, {KTo}, {}}        |2.5039418492439625e-05|
        |GameTreeThreePlayer:0.5:1:11.5|yes|  1  |  2  |{1, 1, 0, 0, 0, 0}|                                  {{A2o}, {76o}, {}, {}, {}, {}}                                   |{0, 0, 0, 0, 0, 0}|           {{}, {}, {}, {}, {}, {}}           |0.0025041322666282256 |
        | GameTreeThreePlayer:0.5:1:12 |yes|  2  |  5  |{1, 2, 1, 0, 0, 1}|                             {{A2o}, {K3o, J8o}, {K4s}, {}, {}, {ATo}}                             |{0, 0, 0, 0, 0, 0}|           {{}, {}, {}, {}, {}, {}}           |0.0024604610987205289 |
        |GameTreeThreePlayer:0.5:1:12.5|yes|  2  |  3  |{1, 0, 2, 0, 0, 0}|                                {{A2o}, {}, {K7o, JTo}, {}, {}, {}}                                |{1, 0, 1, 0, 0, 0}|        {{A2o}, {}, {K7o}, {}, {}, {}}        |0.00097443409925754287|
        | GameTreeThreePlayer:0.5:1:13 |yes|  2  |  5  |{2, 1, 2, 0, 0, 0}|                            {{T7s, 76s}, {Q8o}, {K7o, K5s}, {}, {}, {}}                            |{0, 0, 1, 0, 0, 0}|         {{}, {}, {K7o}, {}, {}, {}}          |0.0021287651376731553 |
        |GameTreeThreePlayer:0.5:1:13.5|yes|  1  |  1  |{0, 0, 1, 0, 0, 0}|                                    {{}, {}, {K8o}, {}, {}, {}}                                    |{0, 0, 1, 0, 0, 0}|         {{}, {}, {K8o}, {}, {}, {}}          |2.2789157163721763e-05|
        | GameTreeThreePlayer:0.5:1:14 |yes|  2  |  3  |{2, 1, 0, 0, 0, 0}|                                {{A4o, K7s}, {K5o}, {}, {}, {}, {}}                                |{0, 0, 0, 0, 0, 0}|           {{}, {}, {}, {}, {}, {}}           |0.0037648978152904355 |
        |GameTreeThreePlayer:0.5:1:14.5|yes|  1  |  3  |{1, 1, 0, 1, 0, 0}|                                 {{A4o}, {87o}, {}, {A8o}, {}, {}}                                 |{0, 0, 0, 0, 0, 0}|           {{}, {}, {}, {}, {}, {}}           | 0.002695817765580838 |
        | GameTreeThreePlayer:0.5:1:15 |yes|  2  |  6  |{1, 2, 2, 1, 0, 0}|                           {{A4o}, {T8o, 87o}, {22, QTo}, {A8o}, {}, {}}                           |{0, 0, 0, 0, 0, 0}|           {{}, {}, {}, {}, {}, {}}           | 0.003091601177018205 |
        |GameTreeThreePlayer:0.5:1:15.5|yes|  3  |  8  |{3, 3, 2, 0, 0, 0}|                     {{A4o, K8s, Q8s}, {K7o, T8o, 87o}, {22, QTo}, {}, {}, {}}                     |{0, 0, 0, 0, 0, 0}|           {{}, {}, {}, {}, {}, {}}           |0.00062032031264086518|
        | GameTreeThreePlayer:0.5:1:16 |yes|  1  |  5  |{1, 1, 1, 1, 0, 1}|                              {{K7s}, {87o}, {Q9s}, {A7s}, {}, {66}}                               |{0, 0, 0, 0, 0, 0}|           {{}, {}, {}, {}, {}, {}}           |0.00099115315679511318|
        |GameTreeThreePlayer:0.5:1:16.5|yes|  2  |  7  |{2, 1, 1, 1, 2, 0}|                         {{KTo, K7s}, {87o}, {A3o}, {A9o}, {A8o, A5s}, {}}                         |{0, 0, 0, 0, 0, 0}|           {{}, {}, {}, {}, {}, {}}           |0.0048397996628728492 |
        | GameTreeThreePlayer:0.5:1:17 |yes|  2  |  6  |{1, 0, 2, 1, 2, 0}|                          {{KTo}, {}, {K9o, JTs}, {A9o}, {A7s, A5s}, {}}                           |{0, 0, 0, 0, 0, 0}|           {{}, {}, {}, {}, {}, {}}           |0.0042421108670099815 |
        |GameTreeThreePlayer:0.5:1:17.5|yes|  2  |  3  |{1, 0, 0, 2, 0, 0}|                                {{KTo}, {}, {}, {A9o, A8s}, {}, {}}                                |{0, 0, 0, 0, 0, 0}|           {{}, {}, {}, {}, {}, {}}           |0.0030190135050972653 |
        | GameTreeThreePlayer:0.5:1:18 |yes|  2  |  4  |{2, 1, 0, 1, 0, 0}|                              {{KTo, JTo}, {K8o}, {}, {A8s}, {}, {}}                               |{0, 0, 0, 0, 0, 0}|           {{}, {}, {}, {}, {}, {}}           |0.0065782208766878014 |
        |GameTreeThreePlayer:0.5:1:18.5|yes|  2  |  3  |{2, 0, 0, 1, 0, 0}|                                {{KTo, JTo}, {}, {}, {KQo}, {}, {}}                                |{0, 0, 0, 0, 0, 0}|           {{}, {}, {}, {}, {}, {}}           |0.0063373804763142538 |

                    btn push/fold

            A    K    Q    J    T    9    8    7    6    5    4    3    2   
          +-----------------------------------------------------------------
        A |18.5 18.5 18.5 18.5 18.5 18.5 18.5 18.5 18.5 18.5 18.5 18.5 18.5
        K |18.5 18.5 18.5 18.5 18.5 18.5 13.5 16.5 12.5 10.0 8.5  7.5  7.0 
        Q |18.5 18.5 18.5 18.5 18.5 18.5 13.5 7.5  7.5  5.5  3.5   0    0  
        J |18.5 18.5 18.5 18.5 18.5 18.5 18.5 8.5   1    0    0    0    0  
        T |18.5 18.5 16.0 18.5 18.5 18.5 18.5 13.0  0    0    0    0    0  
        9 |18.5 9.5  6.5  5.0  4.5  18.5 18.5 13.5  0    0    0    0    0  
        8 |18.5 6.5  4.0  2.5   1    0   18.5 18.0 10.5  0    0    0    0  
        7 |16.0 6.5   0    0    0    0    0   18.5 13.0  0    0    0    0  
        6 |14.0 5.5   0    0    0    0    0    0   18.5 9.5   0    0    0  
        5 |15.5 5.0   0    0    0    0    0    0    0   18.5  0    0    0  
        4 |15.5  0    0    0    0    0    0    0    0    0   18.5  0    0  
        3 |12.5  0    0    0    0    0    0    0    0    0    0   18.5  0  
        2 |12.0  0    0    0    0    0    0    0    0    0    0    0   18.5

                    sb push/fold, given btn fold

            A    K    Q    J    T    9    8    7    6    5    4    3    2   
          +-----------------------------------------------------------------
        A |18.5 18.5 18.5 18.5 18.5 18.5 18.5 18.5 18.5 18.5 18.5 18.5 18.5
        K |18.5 18.5 18.5 18.5 18.5 18.5 18.5 18.5 18.5 18.5 18.5 18.5 18.5
        Q |18.5 18.5 18.5 18.5 18.5 18.5 18.5 18.5 18.5 18.5 14.0 12.5 11.5
        J |18.5 18.5 18.5 18.5 18.5 18.5 18.5 18.5 17.5 14.0 11.5 10.0 8.0 
        T |18.5 18.5 18.5 18.5 18.5 18.5 18.5 18.5 18.5 11.5 10.0 7.0  6.5 
        9 |18.5 18.5 18.5 18.5 18.5 18.5 18.5 18.5 18.5 13.5 6.5  4.0  3.0 
        8 |18.5 18.0 13.0 12.0 15.5 18.5 18.5 18.5 18.5 18.5 10.0 2.5  2.5 
        7 |18.5 15.5 10.0 8.5  8.5  10.5 16.5 18.5 18.5 18.5 13.5 2.0  2.0 
        6 |18.5 14.5 9.5  6.0  5.5  4.5  5.0  10.5 18.5 18.5 14.5 7.0  2.0 
        5 |18.5 14.0 8.5  5.5  4.0  3.0  2.5  2.5  2.0  18.5 18.5 11.5 2.0 
        4 |18.5 12.5 7.5  5.0  3.5  2.5  2.0  2.0  2.0  2.0  18.5 10.0 1.5 
        3 |18.5 12.0 7.0  4.5  3.0  2.0  1.5  1.5  1.5  1.5  1.5  18.5 1.5 
        2 |18.5 11.0 6.5  4.0  2.5  2.0  1.5  1.5  1.5  1.5   1    1   18.5

                    bb call/fold, given btn fold, sb push

            A    K    Q    J    T    9    8    7    6    5    4    3    2   
          +-----------------------------------------------------------------
        A |18.5 18.5 18.5 18.5 18.5 18.5 18.5 18.5 18.5 18.5 18.5 18.5 18.5
        K |18.5 18.5 18.5 18.5 18.5 18.5 16.5 14.5 14.0 13.0 12.0 11.0 10.5
        Q |18.5 18.5 18.5 18.5 18.5 16.0 12.0 10.5 9.5  8.5  8.0  7.5  7.0 
        J |18.5 18.5 18.5 18.5 17.5 12.0 10.5 8.0  6.5  6.5  6.0  5.5  5.0 
        T |18.5 18.5 15.0 12.5 18.5 11.0 9.0  7.0  6.0  5.0  5.0  4.5  4.5 
        9 |18.5 17.0 11.0 9.0  8.0  18.5 8.5  6.5  5.5  4.5  4.0  4.0  3.5 
        8 |18.5 13.0 9.5  7.0  6.5  5.5  18.5 6.0  5.5  4.5  4.0  3.5  3.5 
        7 |18.5 12.0 7.5  5.5  5.0  4.5  4.5  18.5 5.0  4.5  4.0  3.5  3.0 
        6 |18.5 10.5 7.0  5.0  4.5  4.0  4.0  4.0  18.5 4.5  4.0  3.5  3.0 
        5 |18.5 9.5  6.5  4.5  3.5  3.5  3.5  3.5  3.5  18.5 4.5  4.0  3.5 
        4 |18.0 8.5  6.0  4.5  3.5  3.0  3.0  3.0  3.0  3.5  18.5 3.5  3.0 
        3 |16.0 8.5  5.5  4.0  3.5  3.0  2.5  2.5  2.5  3.0  3.0  18.5 3.0 
        2 |15.5 7.5  5.0  4.0  3.0  3.0  2.5  2.5  2.5  2.5  2.5  2.5  15.5

                    sb call/fold, given btn push

            A    K    Q    J    T    9    8    7    6    5    4    3    2  
          +----------------------------------------------------------------
        A |18.5 18.5 18.5 18.5 18.5 18.5 18.0 16.0 13.0 13.0 12.0 10.0 9.5
        K |18.5 18.5 18.5 18.5 13.5 9.0  7.0  6.5  5.5  5.0  4.5  4.5  4.0
        Q |18.5 18.5 18.5 13.0 9.5  6.5  5.0  4.0  3.5  3.5  3.0  3.0  2.5
        J |18.5 10.5 7.5  18.5 8.0  5.5  4.5  3.5  3.0  2.5  2.5  2.5  2.0
        T |18.5 9.0  6.5  5.0  18.5 5.5  4.5  3.5  3.0  2.5  2.0  2.0  2.0
        9 |17.5 6.5  4.0  3.5  3.5  18.5 4.5  3.5  3.0  2.5  2.0  2.0  1.5
        8 |15.0 4.5  3.0  3.0  3.0  3.0  18.5 4.0  3.5  2.5  2.0  1.5  1.5
        7 |11.5 4.0  2.5  2.0  2.0  2.5  2.5  18.5 3.5  3.0  2.5  2.0  1.5
        6 |8.5  3.5  2.0  1.5  2.0  2.0  2.0  2.0  18.5 3.5  2.5  2.0  1.5
        5 |8.5  3.0  2.0  1.5  1.5  1.5  1.5  1.5  2.0  18.5 3.0  2.0  1.5
        4 |7.0  2.5  2.0  1.5  1.5   1    1   1.5  1.5  1.5  15.5 2.0  1.5
        3 |6.5  2.0  1.5  1.5   1    1    1    1    1   1.5   1   12.0 1.5
        2 |6.5  2.0  1.5  1.5   1    1    1    1    1    1    1    1   7.5

                    bb call/fold, given btn push, sb fold

            A    K    Q    J    T    9    8    7    6    5    4    3    2   
          +-----------------------------------------------------------------
        A |18.5 18.5 18.5 18.5 18.5 18.5 18.5 17.0 15.5 17.0 14.0 13.5 13.0
        K |18.5 18.5 18.5 18.5 16.0 11.5 9.0  8.5  7.5  7.5  7.0  6.5  6.5 
        Q |18.5 18.5 18.5 15.0 12.0 8.5  7.5  6.5  5.5  5.5  5.0  5.0  4.5 
        J |18.5 14.5 10.0 18.5 10.5 7.5  6.5  5.5  4.5  4.5  4.5  4.0  4.0 
        T |18.5 10.5 8.0  7.0  18.5 7.5  6.5  5.5  4.5  4.0  4.0  4.0  3.5 
        9 |18.5 8.0  6.0  5.0  5.0  18.5 6.5  5.5  4.5  4.0  3.5  3.5  3.5 
        8 |16.5 6.5  5.0  4.5  4.5  4.5  18.5 5.5  5.0  4.5  4.0  3.5  3.5 
        7 |14.0 6.0  4.5  4.0  3.5  4.0  4.0  18.5 5.0  4.5  4.0  3.5  3.0 
        6 |12.0 5.5  4.0  3.5  3.5  3.5  3.5  4.0  18.5 5.0  4.5  4.0  3.5 
        5 |12.0 5.0  4.0  3.5  3.0  3.0  3.5  3.5  3.5  18.5 4.5  4.0  3.5 
        4 |10.0 5.0  4.0  3.0  3.0  3.0  3.0  3.0  3.5  3.5  18.5 4.0  3.5 
        3 |9.5  5.0  3.5  3.0  3.0  3.0  2.5  3.0  3.0  3.0  3.0  15.0 3.5 
        2 |9.0  4.5  3.5  3.0  3.0  2.5  2.5  2.5  2.5  3.0  3.0  2.5  12.0

                    bb call/fold, given btn push, sb call

            A    K    Q    J    T    9    8    7    6    5    4    3   2  
          +---------------------------------------------------------------
        A |18.5 18.5 18.5 18.5 16.0 11.5 9.5  8.5  7.0  7.5  6.5  6.5 6.5
        K |18.5 18.5 17.5 15.0 12.5 8.5  6.5  6.5  6.0  5.5  5.0  4.5 4.5
        Q |18.5 12.0 18.5 14.0 12.0 8.5  7.0  5.5  5.0  5.0  4.5  4.0 4.0
        J |15.5 8.5  8.0  18.5 12.0 8.5  7.0  6.5  4.5  4.5  4.0  4.0 3.5
        T |12.0 7.0  6.5  6.5  18.5 10.5 7.5  6.5  5.0  4.0  4.0  3.5 3.5
        9 |8.0  5.5  5.0  5.0  5.5  18.5 8.5  7.0  5.5  4.0  3.5  3.5 3.0
        8 |6.5  4.0  4.0  4.0  4.0  4.5  18.5 7.5  6.5  5.0  4.0  3.5 3.0
        7 |6.5  4.0  3.5  3.5  3.5  4.0  4.5  18.5 7.0  6.5  4.5  3.5 3.0
        6 |5.0  3.5  3.0  3.0  3.0  3.0  3.5  4.0  16.0 6.5  5.0  4.0 3.5
        5 |5.5  3.5  3.0  3.0  2.5  2.5  3.0  3.5  4.0  12.5 6.5  4.5 4.0
        4 |5.0  3.0  3.0  3.0  2.5  2.5  2.5  3.0  3.0  3.5  10.5 4.5 3.5
        3 |4.5  3.0  3.0  2.5  2.5  2.5  2.0  2.5  2.5  3.0  3.0  8.0 3.5
        2 |4.5  3.0  2.5  2.5  2.5  2.0  2.0  2.0  2.5  2.5  2.5  2.5 6.5

