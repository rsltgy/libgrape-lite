//
// Created by rsltgy on 20/12/2020.
//

#ifndef LIBGRAPE_LITE_APAIR_H
#define LIBGRAPE_LITE_APAIR_H


#include <grape/grape.h>
#include "../ParaMatch/ParaMatch.h"
#include "Apair_context.h"
#include "grape/utils/vertex_array.h"
#include "../KDTree/KDTree.h"

using namespace std;


namespace grape{

    template <typename FRAG_T>
    class Apair : public ParallelAppBase<FRAG_T, ApairContext<FRAG_T>>,
                  public ParallelEngine {



// specialize the templated worker.
    INSTALL_PARALLEL_WORKER(Apair<FRAG_T>, ApairContext<FRAG_T>, FRAG_T)
        using vertex_t = typename fragment_t::vertex_t;


        static constexpr MessageStrategy message_strategy = MessageStrategy::kAlongIncomingEdgeToOuterVertex;
        static constexpr LoadStrategy load_strategy = LoadStrategy::kBothOutIn;


        void PEval(const fragment_t& frag, context_t& ctx,
                   message_manager_t& messages) {

            auto &rev = ctx.rev;
            auto &match_set = ctx.match_set;
            auto &C = ctx.C;
            auto &GD = ctx.GD;
            auto &cache = ctx.cache;
            auto &ecache_u = ctx.ecache_u;
            auto &ecache_v = ctx.ecache_v;
            auto &g_paths = ctx.g_paths;
            auto &g_descendants = ctx.g_descendants;
            auto &word_embeddings = ctx.word_embeddings;
            auto &message_cache = ctx.message_cache;
            auto &message_address = ctx.message_address;
            auto &sigma = ctx.sigma;
            auto &delta = ctx.delta;
            messages.InitChannels(thread_num(), 2 * 1023 * 64, 2 * 1024 * 64);

            cout << "Candidate generation started at Frag " << frag.fid() <<  endl;
            timer_next("Calculate word vector");

            auto inner_vert = frag.InnerVertices();
            {
              pointVec points;

              points.resize(inner_vert.size());

              ForEach(inner_vert, [&](int tid, vertex_t v) {
                string v_str = frag.GetData(v);
                vector<double> v_g_word_vector;
                boost::trim_right(v_str);
                Reader::calculate_word_vector(ctx.word_embeddings, v_str,
                                              v_g_word_vector);
                if (!v_g_word_vector.empty()) {
                  auto oid = frag.GetId(v);
                  points[v.GetValue()] = std::make_pair(oid, v_g_word_vector);
                }
              });

              timer_next("Candidate Generation");

              KDTree tree(points);

              for(int u_t = 0; u_t < GD.number_of_nodes(); u_t++){
                  if(!GD.nodes()[u_t].empty()){
                      vector<double> u_t_word_vector;
                      Reader::calculate_word_vector(word_embeddings,GD.nodes()[u_t],u_t_word_vector);
                      if(!u_t_word_vector.empty()) {
                          point_t t = make_pair(u_t,u_t_word_vector);
                          auto NNs = tree.neighborhood_points(t, sigma);
                          for(auto returned_match : NNs){
                              C.emplace_back(u_t, make_pair(returned_match.first, u_t));
                          }
                      }
                  }
              }
            }
            cout << "Candidate generation ended at Frag " << frag.fid() << " with candidate size " << C.size() <<   endl;

            timer_next("Parametric Simulation from Candidates");
            sort(C.begin(),C.end(),[](const pair<int,pair<int,int>> &a, const pair<int,pair<int,int>> b){
                return a.second.second < b.second.second;
            });

            bool match = false;
            for(auto& v : C){
                auto pair = make_pair(v.first,v.second.first);
                auto it = cache.find(pair);
                if(it != ctx.cache.end() && cache[pair].first  ){
                    match_set[v.second.first].insert(v.first);
                }
                else{
                    ParaMatch<FRAG_T> p;
                    match = p.match_pair(GD,frag,g_paths,g_descendants,v.first,v.second.first,sigma,delta,cache,word_embeddings,ecache_u,ecache_v,rev);
                    if(match)
                        match_set[v.second.first].insert(v.first);
                }
            }
            for (auto &ca : cache) {
                message_cache[ca.first.second].push_back(ca.first.first);
            }
            auto inner_vertices = frag.InnerVertices();
            auto outer_vertices = frag.OuterVertices();

            timer_next("Message Address Finding");

            for(auto o_v : outer_vertices){
                auto o_v_oid = frag.GetId(o_v);
                auto i_e = frag.GetIncomingAdjList(o_v);
                for(auto e : i_e){
                    auto i_i_v = e.get_neighbor();
                    auto i_v_oid = frag.GetId(i_i_v);
                    //cout << i_v_oid << " is ancestor of " << o_v_oid << endl;
                    message_address[o_v_oid].push_back(i_v_oid);
                }
            }

          ForEach(inner_vertices, [&](int tid, vertex_t i_v) {
            auto oid = frag.GetId(i_v);
            if(frag.IsIncomingBorderVertex(i_v)){
              if(match_set.find(oid) != match_set.end()){
                messages.Channels()[tid].SendMsgThroughIEdges(frag,i_v,match_set[oid]);
              }
            }
            });
        }


        void IncEval(const fragment_t& frag, context_t& ctx,
                     message_manager_t& messages) {
            auto &rev = ctx.rev;
            auto &GD = ctx.GD;
            auto &cache = ctx.cache;
            auto &ecache_u = ctx.ecache_u;
            auto &ecache_v = ctx.ecache_v;
            auto &g_paths = ctx.g_paths;
            auto &g_descendants = ctx.g_descendants;
            auto &word_embeddings = ctx.word_embeddings;
            double &sigma = ctx.sigma;
            double &delta = ctx.delta;
            auto &match_set = ctx.match_set;
            auto &message_cache = ctx.message_cache;
            auto &message_address = ctx.message_address;
            auto &messages_received = ctx.messages_received;
            auto &channel_0 = messages.Channels()[0];

            timer_next("IncEval");
            messages.ParallelProcess<fragment_t, set<int>>(
                    1, frag, [&frag,&messages_received](int tid, vertex_t v, set<int>& msg) {
                        // auto v_received =  frag.GetId(v);
                        // cout << frag.fid() << " received " << v_received <<  endl;
                        messages_received[v].push_back(msg);
                    });


            for(auto vv : frag.InnerVertices()) {
              auto &message = messages_received[vv];
              if(!message.empty()) {
                auto message_come_from = frag.GetId(vv);
                std::pair<int,vector<int>> msg;
                auto all_v_s = message_address[message_come_from];
                for(unsigned int v : all_v_s){
                    msg.first = v;
                    vector<int> v_s = message_cache[v];
                    for(int u : v_s){
                        auto u_v = make_pair(u, v);
                        pair<bool, vector<pair<int, int>>> match_result_and_witnesses = cache[u_v];
                        if(!match_result_and_witnesses.first){
                            auto witness_vertices =  match_result_and_witnesses.second;
                            double sum = 0;
                            ParaMatch<FRAG_T> p;
                            for(auto& wit : witness_vertices){
                                double local_sum = p.calculate_path_similarity(GD,g_paths,ctx.word_embeddings, u, wit.first, v, wit.second);
                                sum += local_sum;
                            }

                            for(auto & v_prime_and_all_u_primes : message ){
                                unsigned int v_prime = message_come_from;
                                cout << "v  " << v << " v prime " << v_prime << endl;
                                for(auto u_prime : v_prime_and_all_u_primes){
                                    double local_sum = p.calculate_path_similarity(GD,g_paths,ctx.word_embeddings, u, u_prime , v, v_prime);
                                    sum += local_sum;
                                    if (sum >= ctx.delta)  {
                                        match_set[u].insert(v);
                                        msg.second.push_back(u);
                                        cache[u_v].first = false;

                                        for(const auto &u_p_v_p: rev[u_v]) {
                                            cache.erase(u_p_v_p);
                                        }

                                        for(const auto &u_p_v_p: rev[u_v]) {
                                            p.match_pair(GD, frag, g_paths,g_descendants,u_p_v_p.first, u_p_v_p.second, sigma, delta, cache, word_embeddings, ecache_u, ecache_v, rev);
                                        }
                                        rev[u_v].clear();
                                        break;
                                    }
                                }
                                if (sum >= ctx.delta)
                                    break;
                            }
                        }
                    }


                    if(!msg.second.empty()){
                        auto inner_vertices = frag.InnerVertices();
                        for(auto i_v : inner_vertices){
                            auto oid = frag.GetId(i_v);
                            if(frag.IsIncomingBorderVertex(i_v) && v == oid){
                                if(match_set.find(oid) != match_set.end()) {
                                    channel_0.SendMsgThroughIEdges(frag, i_v, std::make_pair(oid, match_set[oid]));
                                }
                            }
                        }
                    }

                }

              }
              message.clear();
            }

        }
    };



}


#endif  // LIBGRAPE_LITE_APAIR_H
