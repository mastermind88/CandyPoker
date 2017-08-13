#ifndef PS_SUPPORT_SINGLETON_FACTORY_H
#define PS_SUPPORT_SINGLETON_FACTORY_H

#include <boost/type_index.hpp>
#include <boost/exception/all.hpp>
#include <iostream>
#include <map>
#include <functional>
#include <memory>
#include <mutex>

namespace ps{
namespace support{

        template<class T>
        struct singleton_factory{
        private:
                using maker_t = 
                        std::function<
                                std::unique_ptr<T>()
                        >;
                struct item{
                        explicit item(maker_t maker)
                                :maker_(std::move(maker))
                        {}
                        T& get(){
                                if( ! ptr_ ){
                                        std::lock_guard<std::mutex> lock(mtx_);
                                        if( ! ptr_ ){
                                                ptr_ = maker_();
                                        }
                                }
                                return *ptr_;
                        }
                private:
                        std::unique_ptr<T> ptr_;
                        maker_t maker_;
                        std::mutex mtx_;
                };
                template<class U>
                void register_impl( std::string const& name ){
                        m_.emplace(name, std::make_unique<item>([](){ return std::make_unique<U>(); }));
                }
                T& get_impl(std::string const& name){
                        auto iter = m_.find(name);
                        if( iter == m_.end() )
                                BOOST_THROW_EXCEPTION(std::domain_error("singlton " + name + " not registered"));
                        return iter->second->get();
                }
        public:

                static singleton_factory* get_inst(){
                        static singleton_factory* mem = nullptr;
                        if( mem == nullptr )
                                mem = new singleton_factory;
                        return mem;
                }

                template<class U>
                static void register_( std::string const& name  = "__default__"){
                        get_inst()->template register_impl<U>(name);
                }
                static T& get(std::string const& name = "__default__"){
                        std::cout << "getting name = " << name << " for " << boost::typeindex::type_id<T>().pretty_name() << "\n";
                        return get_inst()->get_impl(name);
                }

                
        private:
                std::map<
                        std::string,
                        std::unique_ptr<item>
                > m_;
        };
} // support
} // ps
#endif // PS_SUPPORT_SINGLETON_FACTORY_H
