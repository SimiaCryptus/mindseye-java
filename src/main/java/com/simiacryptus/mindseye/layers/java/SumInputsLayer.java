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
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;
import org.jetbrains.annotations.NotNull;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;
import java.util.function.DoubleUnaryOperator;
import java.util.function.IntFunction;

import static com.simiacryptus.mindseye.lang.Result.anyAlive;

@SuppressWarnings("serial")
public class SumInputsLayer extends LayerBase {

  public SumInputsLayer() {
  }

  protected SumInputsLayer(@Nonnull final JsonObject id) {
    super(id);
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static SumInputsLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new SumInputsLayer(json);
  }

  public static PipelineNetwork combine(@Nullable PipelineNetwork... networks) {
    return PipelineNetwork.combine(new SumInputsLayer(), networks);
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    TensorList data = fwd(RefUtil.addRefs(inObj));
    boolean alive = anyAlive(RefUtil.addRefs(inObj));
    Accumulator accumulator = new Accumulator(inObj);
    return new Result(data, accumulator, alive);
  }

  @NotNull
  private TensorList fwd(@Nonnull Result[] inObj) {
    return RefUtil.get(RefArrays.stream(inObj).parallel().map(x -> {
      return Result.getData(x);
    }).reduce((l, r) -> {
      assert l.length() == r.length() || 1 == l.length() || 1 == r.length();
      return new TensorArray(RefIntStream.range(0, l.length()).parallel()
          .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) i -> {
            @Nullable final Tensor left = l.get(1 == l.length() ? 0 : i);
            @Nullable final Tensor right = r.get(1 == r.length() ? 0 : i);
            @Nullable
            Tensor tensor = null;
            if (right.length() == 1) {
              RefUtil.freeRef(tensor);
              tensor = left.mapParallel(RefUtil.wrapInterface((DoubleUnaryOperator) v -> v + right.get(0),
                  right.addRef()));
            } else {
              RefUtil.freeRef(tensor);
              tensor = left.reduceParallel(right.addRef(), Double::sum);
            }
            right.freeRef();
            left.freeRef();
            return tensor;
          }, r, l)).toArray(Tensor[]::new));
    }));
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    return super.getJsonStub();
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return RefArrays.asList();
  }

  public @SuppressWarnings("unused")
  void _free() {
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  SumInputsLayer addRef() {
    return (SumInputsLayer) super.addRef();
  }

  private static class Accumulator extends Result.Accumulator {

    private final Result[] inObj;

    public Accumulator(Result... inObj) {
      this.inObj = inObj;
    }

    @Override
    public void accept(@Nullable DeltaSet<UUID> buffer, @Nullable TensorList delta) {
      for (@Nonnull final Result input : inObj) {
        if (input.isAlive()) {
          @Nonnull
          TensorList projectedDelta = delta == null ? null : delta.addRef();
          TensorList temp_55_0007 = input.getData();
          assert projectedDelta != null;
          if (1 < projectedDelta.length() && temp_55_0007.length() == 1) {
            TensorArray projectedDelta1 = new TensorArray(RefUtil.get(projectedDelta.stream().parallel().reduce((a, b) -> {
              return Tensor.add(a, b);
            })));
            projectedDelta.freeRef();
            projectedDelta = projectedDelta1;
          }
          temp_55_0007.freeRef();
          TensorList temp_55_0008 = input.getData();
          if (1 < Tensor.length(projectedDelta.getDimensions())
              && Tensor.length(temp_55_0008.getDimensions()) == 1) {
            @Nonnull
            TensorArray new_projectedDelta = new TensorArray(projectedDelta.stream().map(t -> {
              Tensor temp_55_0004 = new Tensor(new double[]{t.sum()});
              t.freeRef();
              return temp_55_0004;
            }).toArray(Tensor[]::new));
            projectedDelta.freeRef();
            projectedDelta = new_projectedDelta;
          }
          temp_55_0008.freeRef();
          DeltaSet<UUID> buffer1 = buffer == null ? null : buffer.addRef();
          Result.Accumulator accumulator = input.getAccumulator();
          try {
            accumulator.accept(buffer1, projectedDelta);
          } finally {
            accumulator.freeRef();
          }
        }
      }
      if (null != delta)
        delta.freeRef();
      if (null != buffer)
        buffer.freeRef();
    }

    public @SuppressWarnings("unused")
    void _free() {
      super._free();
      RefUtil.freeRef(inObj);
    }
  }
}
