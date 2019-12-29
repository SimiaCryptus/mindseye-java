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
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.network.PipelineNetwork;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.stream.IntStream;

@SuppressWarnings("serial")
public class SumInputsLayer extends LayerBase {

  public SumInputsLayer() {
  }

  protected SumInputsLayer(@Nonnull final JsonObject id) {
    super(id);
  }

  @SuppressWarnings("unused")
  public static SumInputsLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new SumInputsLayer(json);
  }

  public static PipelineNetwork combine(PipelineNetwork... networks) {
    return PipelineNetwork.combine(new SumInputsLayer(), networks);
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    return new Result(Arrays.stream(inObj).parallel().map(x -> {
      return x.getData();
    }).reduce((l, r) -> {
      assert l.length() == r.length() || 1 == l.length() || 1 == r.length();
      return new TensorArray(IntStream.range(0, l.length()).parallel().mapToObj(i -> {
            @Nullable final Tensor left = l.get(1 == l.length() ? 0 : i);
            @Nullable final Tensor right = r.get(1 == r.length() ? 0 : i);
            @Nullable
            Tensor tensor;
            if (right.length() == 1) {
              tensor = left.mapParallel(v -> v + right.get(0));
            } else {
              tensor = left.reduceParallel(right, (v1, v2) -> v1 + v2);
            }
            return tensor;
          }).toArray(i -> new Tensor[i]));
    }).get(), (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList delta) -> {
      for (@Nonnull final Result input : inObj) {
        if (input.isAlive()) {
          @Nonnull
          TensorList projectedDelta = delta;
          if (1 < projectedDelta.length() && input.getData().length() == 1) {
            projectedDelta = new TensorArray(projectedDelta.stream().parallel().reduce((a, b) -> {
                      return a.addAndFree(b);
                    }).get());
          }
          if (1 < Tensor.length(projectedDelta.getDimensions())
              && Tensor.length(input.getData().getDimensions()) == 1) {
            @Nonnull
            TensorArray new_projectedDelta = new TensorArray(projectedDelta.stream().map(t -> {
              return new Tensor(new double[]{t.sum()});
            }).toArray(i -> new Tensor[i]));
            projectedDelta = new_projectedDelta;
          }
          input.accumulate(buffer, projectedDelta);
        }
      }
    }) {

      @Override
      public boolean isAlive() {
        for (@Nonnull final Result element : inObj)
          if (element.isAlive()) {
            return true;
          }
        return false;
      }

      @Override
      protected void _free() {
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
