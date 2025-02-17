#ifndef LIB_EVAL_RANK_OPT_DEVICE_H
#define LIB_EVAL_RANK_OPT_DEVICE_H


namespace ps{

        
struct rank_opt_item{
        holdem_id hid;
        rank_id r0;
        suit_id s0;
        rank_id r1; 
        suit_id s1;
};
struct rank_opt_device : std::vector<rank_opt_item>{

        #if 0
        struct segmented{
                std::vector<rank_opt_item> segment_0;
                std::vector<rank_opt_item> segment_1;
                std::vector<rank_opt_item> segment_2;
        };
        std::array<segmented, 4> segments;
        #endif

        template<class Con>
        static rank_opt_device create(Con const& con){
                rank_opt_device result;
                result.resize(con.size());
                rank_opt_item* out = &result[0];
                size_t index = 0;
                for(auto hid : con){
                        auto const& hand{holdem_hand_decl::get(hid)};

                        std::uint16_t nfnp_mask = static_cast<std::uint16_t>(1) << hand.first().rank().id() |
                                                  static_cast<std::uint16_t>(1) << hand.second().rank().id();
                        if( detail::popcount(nfnp_mask) != 2 ){
                                nfnp_mask = ~static_cast<std::uint16_t>(1);
                        }

                        rank_opt_item item{
                                hid,
                                hand.first().rank().id(),
                                hand.first().suit().id(),
                                hand.second().rank().id(),
                                hand.second().suit().id()
                        };
                        *out = item;
                        ++out;
                        ++index;
                }

                #if 0
                for(unsigned sid=0;sid!=4;++sid){
                        auto& seg = result.segments[sid];
                        for(auto const& _ : result){
                                unsigned count = 0;
                                bool s0c = ( _.s0 == sid );
                                bool s1c = ( _.s1 == sid );

                                if( s0c ) ++count;
                                if( s1c ) ++count;


                                switch(count){
                                case 0:
                                        seg.segment_0.push_back(_);
                                        break;
                                case 1:
                                        seg.segment_1.push_back(_);
                                        if( s1c ){
                                                auto& obj = seg.segment_1.back();
                                                std::swap(obj.c0        , obj.c1);
                                                std::swap(obj.r0        , obj.r1);
                                                std::swap(obj.s0        , obj.s1);
                                                std::swap(obj.r0_shifted, obj.r1_shifted);
                                        }
                                        break;
                                case 2:
                                        seg.segment_2.push_back(_);
                                        break;
                                }
                        }
                        std::cout << "rod.size() => " << result.size() << "\n"; // __CandyPrint__(cxx-print-scalar,rod.size())
                        std::cout << "(seg.segment_0.size()+ seg.segment_1.size()+ seg.segment_2.size()) => " << (seg.segment_0.size()+ seg.segment_1.size()+ seg.segment_2.size()) << "\n"; // __CandyPrint__(cxx-print-scalar,(seg.segment_0.size()+ seg.segment_1.size()+ seg.segment_2.size()))
                }
                #endif

                return result;
        }
};

} // end namespace ps

#endif // LIB_EVAL_RANK_OPT_DEVICE_H
