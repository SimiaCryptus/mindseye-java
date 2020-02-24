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
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefArrayList;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;
import java.util.function.IntFunction;

@SuppressWarnings("serial")
public class EntropyLossLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(EntropyLossLayer.class);

  public EntropyLossLayer() {
  }

  protected EntropyLossLayer(@Nonnull final JsonObject id) {
    super(id);
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static EntropyLossLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new EntropyLossLayer(json);
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {

    final double zero_tol = 1e-12;
    TensorList indata = inObj[0].getData();
    @Nonnull final RefArrayList<Tensor> gradient = new RefArrayList<>();
    final double max_prob = 1.;
    boolean alive = alive(inObj[0].addRef());
    TensorArray data = fwd(zero_tol, indata.addRef(), gradient.addRef(), max_prob, RefUtil.addRefs(inObj));
    final Result.Accumulator accumulator1 = inObj[0].getAccumulator();
    final boolean alive1 = inObj[0].isAlive();
    final Result.Accumulator accumulator2 = inObj[1].getAccumulator();
    final boolean alive2 = inObj[1].isAlive();
    RefUtil.freeRef(inObj);
    Result.Accumulator accumulator = new Accumulator(indata, gradient, max_prob, zero_tol, accumulator1, alive1, accumulator2, alive2);
    return new Result(data, accumulator, alive);
  }

  private boolean alive(Result result) {
    try {
      if (result.isAlive()) return true;
      else return false;
    } finally {
      result.freeRef();
    }
  }

  @NotNull
  private TensorArray fwd(double zero_tol, TensorList indata, RefArrayList<Tensor> gradient, double max_prob, @Nonnull Result[] inObj) {
    return new TensorArray(RefIntStream.range(0, indata.length())
            .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
              @Nullable final Tensor l = indata.get(dataIndex);
              TensorList temp_03_0006 = inObj[1].getData();
              @Nullable final Tensor r = temp_03_0006.get(dataIndex);
              temp_03_0006.freeRef();
              if (l.length() != r.length()) {
                IllegalArgumentException temp_03_0004 = new IllegalArgumentException(
                    l.length() + " != " + r.length());
                l.freeRef();
                r.freeRef();
                throw temp_03_0004;
              }
              @Nonnull final Tensor gradientTensor = new Tensor(l.getDimensions());
              @Nullable final double[] gradientData = gradientTensor.getData();
              double total = 0;
              @Nullable final double[] ld = l.getData();
              @Nullable final double[] rd = r.getData();
              r.freeRef();
              for (int i = 0; i < l.length(); i++) {
                final double lv = Math.max(Math.min(ld[i], max_prob), zero_tol);
                final double rv = rd[i];
                if (rv > 0) {
                  gradientData[i] = -rv / lv;
                  total += -rv * Math.log(lv);
                } else {
                  gradientData[i] = 0;
                }
              }
              l.freeRef();
              //assert total >= 0;
              gradient.add(dataIndex, gradientTensor);
              //RefUtil.set(gradient, dataIndex, gradientTensor);
              return new Tensor(new double[]{total}, 1);
            }, indata, inObj, gradient)).toArray(Tensor[]::new));
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
  EntropyLossLayer addRef() {
    return (EntropyLossLayer) super.addRef();
  }

  private static class Accumulator extends Result.Accumulator {

    private final TensorList indata;
    private final RefArrayList<Tensor> gradient;
    private final double max_prob;
    private final double zero_tol;
    private Result.Accumulator accumulator1;
    private Result.Accumulator accumulator0;
    private boolean alive0;
    private boolean alive1;

    public Accumulator(TensorList indata, RefArrayList<Tensor> gradient, double max_prob, double zero_tol, Result.Accumulator accumulator0, boolean alive0, Result.Accumulator accumulator1, boolean alive1) {
      this.indata = indata;
      this.gradient = gradient;
      this.max_prob = max_prob;
      this.zero_tol = zero_tol;
      this.accumulator1 = accumulator1;
      this.accumulator0 = accumulator0;
      this.alive0 = alive0;
      this.alive1 = alive1;
    }

    @Override
    public void accept(@Nullable DeltaSet<UUID> buffer, @Nonnull TensorList delta) {
      if (alive1) {
        @Nonnull
        TensorArray tensorArray = new TensorArray(RefIntStream.range(0, delta.length())
            .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
                  Tensor deltaTensor = delta.get(dataIndex);
                  @Nullable final Tensor inputTensor = indata.get(dataIndex);
                  Tensor tensor = gradient.get(dataIndex);
                  @Nonnull final Tensor passback = new Tensor(tensor.getDimensions());
                  tensor.freeRef();
                  for (int i = 0; i < passback.length(); i++) {
                    final double lv = Math.max(Math.min(inputTensor.get(i), max_prob), zero_tol);
                    final double value = -deltaTensor.get(0) * Math.log(lv);
                    if (Double.isFinite(value)) passback.set(i, value);
                  }
                  inputTensor.freeRef();
                  deltaTensor.freeRef();
                  return passback;
                }, indata.addRef(), RefUtil.addRef(gradient),
                delta.addRef()))
            .toArray(Tensor[]::new));
        DeltaSet<UUID> buffer1 = buffer == null ? null : buffer.addRef();
        accumulator1.accept(buffer1, tensorArray);
      }
      if (alive0) {
        @Nonnull
        TensorArray tensorArray = new TensorArray(RefIntStream.range(0, delta.length())
            .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
              Tensor tensor = delta.get(dataIndex);
              Tensor tensor1 = gradient.get(dataIndex);
              @Nonnull final Tensor passback = new Tensor(tensor1.getDimensions());
              for (int i = 0; i < passback.length(); i++) {
                passback.set(i, tensor.get(0) * tensor1.get(i));
              }
              tensor1.freeRef();
              tensor.freeRef();
              return passback;
            }, delta.addRef(), RefUtil.addRef(gradient)))
            .toArray(Tensor[]::new));
        DeltaSet<UUID> buffer1 = buffer == null ? null : buffer.addRef();
        accumulator0.accept(buffer1, tensorArray);
      }
      delta.freeRef();
      if (null != buffer)
        buffer.freeRef();
    }

    public @SuppressWarnings("unused")
    void _free() {
      super._free();
      accumulator0.freeRef();
      accumulator1.freeRef();
      indata.freeRef();
      RefUtil.freeRef(gradient);
    }
  }
}
