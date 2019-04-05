/*
 * Copyright (c) 2019 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.layers.java;

import com.google.gson.JsonObject;
import com.simiacryptus.lang.ref.ReferenceCountingBase;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.InnerNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.stream.IntStream;

/**
 * Sums all inputs together, element-by-element, assuming they all have the same dimension.
 */
@SuppressWarnings("serial")
public class SumInputsLayer extends LayerBase {

  /**
   * Instantiates a new Sum inputs key.
   */
  public SumInputsLayer() {
  }

  /**
   * Instantiates a new Sum inputs key.
   *
   * @param id the id
   */
  protected SumInputsLayer(@Nonnull final JsonObject id) {
    super(id);
  }

  /**
   * From json sum inputs key.
   *
   * @param json the json
   * @param rs   the rs
   * @return the sum inputs key
   */
  public static SumInputsLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new SumInputsLayer(json);
  }

  public static PipelineNetwork combine(PipelineNetwork... networks) {
    Arrays.stream(networks).forEach(ReferenceCountingBase::assertAlive);
    PipelineNetwork pipelineNetwork = new PipelineNetwork(1);
    pipelineNetwork.wrap(new SumInputsLayer(), Arrays.stream(networks).map(network -> {
      InnerNode node = transferNode(pipelineNetwork, network.getHead());
      network.freeRef();
      return node;
    }).toArray(i -> new DAGNode[i]));
    return pipelineNetwork;
  }

  public static InnerNode transferNode(PipelineNetwork pipelineNetwork, DAGNode head) {
    return pipelineNetwork.add(head.getLayer(), Arrays.stream(head.getInputs()).map(input -> {
      if (input.getNetwork().inputNodes.containsKey(input.getId())) {
        return pipelineNetwork.getInput(input.getNetwork().inputHandles.indexOf(input.getId()));
      } else {
        Layer inputLayer = input.getLayer();
        if (inputLayer == null) throw new IllegalArgumentException(input.getClass().toString());
        return pipelineNetwork.getNodes().stream().filter(dagNode -> {
          Layer layer = dagNode.getLayer();
          return null != layer && layer.getId().equals(inputLayer.getId());
        }).findFirst().orElseGet(() -> {
          return transferNode(pipelineNetwork, input);
        });
      }
    }).toArray(i -> new DAGNode[i]));
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    Arrays.stream(inObj).forEach(nnResult -> nnResult.addRef());
    Arrays.stream(inObj).forEach(x -> x.getData().addRef());
    return new Result(Arrays.stream(inObj).parallel().map(x -> {
      TensorList data = x.getData();
      data.addRef();
      return data;
    }).reduce((l, r) -> {
      assert l.length() == r.length() || 1 == l.length() || 1 == r.length();
      @Nonnull TensorArray sum = TensorArray.wrap(IntStream.range(0, l.length()).parallel()
          .mapToObj(i -> {
            @Nullable final Tensor left = l.get(1 == l.length() ? 0 : i);
            @Nullable final Tensor right = r.get(1 == r.length() ? 0 : i);
            @Nullable Tensor tensor;
            if (right.length() == 1) {
              tensor = left.mapParallel(v -> v + right.get(0));
            } else {
              tensor = left.reduceParallel(right, (v1, v2) -> v1 + v2);
            }
            left.freeRef();
            right.freeRef();
            return tensor;
          })
          .toArray(i -> new Tensor[i]));
      l.freeRef();
      r.freeRef();
      return sum;
    }).get(), (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList delta) -> {
      try {
        for (@Nonnull final Result input : inObj) {
          if (input.isAlive()) {
            delta.addRef();
            @Nonnull TensorList projectedDelta = delta;
            if (1 < projectedDelta.length() && input.getData().length() == 1) {
              TensorArray new_projectedDelta = TensorArray.wrap(projectedDelta.stream().parallel().reduce((a, b) -> {
                @Nullable Tensor c = a.addAndFree(b);
                b.freeRef();
                return c;
              }).get());
              projectedDelta.freeRef();
              projectedDelta = new_projectedDelta;
            }
            if (1 < Tensor.length(projectedDelta.getDimensions()) && Tensor.length(input.getData().getDimensions()) == 1) {
              @Nonnull TensorArray new_projectedDelta = TensorArray.wrap(projectedDelta.stream().map(t -> {
                Tensor tensor = new Tensor(new double[]{t.sum()});
                t.freeRef();
                return tensor;
              }).toArray(i -> new Tensor[i]));
              projectedDelta.freeRef();
              projectedDelta = new_projectedDelta;
            }
            input.accumulate(buffer, projectedDelta);
          }
        }
      } finally {
        delta.freeRef();
      }
    }) {

      @Override
      protected void _free() {
        Arrays.stream(inObj).forEach(nnResult -> nnResult.freeRef());
        Arrays.stream(inObj).forEach(x -> x.getData().freeRef());
      }

      @Override
      public boolean isAlive() {
        for (@Nonnull final Result element : inObj)
          if (element.isAlive()) {
            return true;
          }
        return false;
      }

    };
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    return super.getJsonStub();
  }

  @Nonnull
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }

}
