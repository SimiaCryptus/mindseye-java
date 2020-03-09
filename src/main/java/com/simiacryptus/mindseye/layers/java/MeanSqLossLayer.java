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
import com.simiacryptus.ref.lang.RefIgnore;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.*;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;
import java.util.function.IntFunction;

@SuppressWarnings("serial")
public class MeanSqLossLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(MeanSqLossLayer.class);

  public MeanSqLossLayer() {
  }

  protected MeanSqLossLayer(@Nonnull final JsonObject id) {
    super(id);
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static MeanSqLossLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new MeanSqLossLayer(json);
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    if (2 != inObj.length) {
      RefUtil.freeRef(inObj);
      throw new IllegalArgumentException();
    }
    TensorList temp_20_0011 = inObj[0].getData();
    final int leftLength = temp_20_0011.length();
    temp_20_0011.freeRef();
    TensorList temp_20_0012 = inObj[1].getData();
    final int rightLength = temp_20_0012.length();
    temp_20_0012.freeRef();
    if (leftLength != rightLength && leftLength != 1 && rightLength != 1) {
      RefUtil.freeRef(inObj);
      throw new IllegalArgumentException(leftLength + " != " + rightLength);
    }
    @Nonnull final Tensor diffs[] = new Tensor[leftLength];
    try {
      TensorArray data = fwd(leftLength, rightLength, diffs, RefUtil.addRef(inObj));
      boolean alive = inObj[0].isAlive() || inObj[1].isAlive();
      final Result.Accumulator accumulator1 = inObj[0].getAccumulator();
      final Result.Accumulator accumulator2 = inObj[1].getAccumulator();
      final boolean alive1 = inObj[0].isAlive();
      final boolean alive2 = inObj[1].isAlive();
      RefUtil.freeRef(inObj);
      Result.Accumulator accumulator = new Accumulator(diffs, leftLength, rightLength, accumulator1, accumulator2, alive1, alive2);
      return new Result(data, accumulator, alive);
    } finally {
    }
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
  MeanSqLossLayer addRef() {
    return (MeanSqLossLayer) super.addRef();
  }

  @NotNull
  private TensorArray fwd(int leftLength, int rightLength, @RefIgnore Tensor[] diffs, @Nonnull Result[] inObj) {
    return new TensorArray(RefIntStream.range(0, leftLength)
        .mapToObj(RefUtil.wrapInterface((IntFunction<Tensor>) dataIndex -> {
              TensorList temp_20_0013 = inObj[0].getData();
              @Nullable final Tensor a = temp_20_0013.get(1 == leftLength ? 0 : dataIndex);
              temp_20_0013.freeRef();
              TensorList temp_20_0014 = inObj[1].getData();
              @Nullable final Tensor b = temp_20_0014.get(1 == rightLength ? 0 : dataIndex);
              temp_20_0014.freeRef();
              if (a.length() != b.length()) {
                IllegalArgumentException temp_20_0003 = new IllegalArgumentException(RefString.format("%s != %s",
                    RefArrays.toString(a.getDimensions()), RefArrays.toString(b.getDimensions())));
                a.freeRef();
                b.freeRef();
                throw temp_20_0003;
              }
              @Nonnull final Tensor r = a.minus(b.addRef());
              b.freeRef();
              a.freeRef();
              Tensor tensor = new Tensor(new double[]{r.sumSq() / r.length()}, 1);
              RefUtil.set(diffs, dataIndex, r);
              return tensor;
            }, inObj)
        ).toArray(Tensor[]::new));
  }

  private static class Accumulator extends Result.Accumulator {

    private final Tensor[] diffs;
    private final int leftLength;
    private final int rightLength;
    private Result.Accumulator accumulator0;
    private Result.Accumulator accumulator1;
    private boolean alive0;
    private boolean alive1;

    public Accumulator(Tensor[] diffs, int leftLength, int rightLength, Result.Accumulator accumulator0, Result.Accumulator accumulator1, boolean alive0, boolean alive1) {
      this.diffs = diffs;
      this.leftLength = leftLength;
      this.rightLength = rightLength;
      this.accumulator0 = accumulator0;
      this.accumulator1 = accumulator1;
      this.alive0 = alive0;
      this.alive1 = alive1;
    }

    @Override
    public void accept(@Nullable DeltaSet<UUID> buffer, @Nonnull TensorList data) {
      if (alive0) {
        RefList<Tensor> temp_20_0015 = RefIntStream.range(0, data.length()).parallel()
            .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
              @Nullable
              Tensor tensor = data.get(dataIndex);
              Tensor diff = diffs[dataIndex].addRef();
              Tensor temp_20_0005 = diff.scale(tensor.get(0) * 2.0 / diff.length());
              diff.freeRef();
              tensor.freeRef();
              return temp_20_0005;
            }, data.addRef(), RefUtil.addRef(diffs))).collect(RefCollectors.toList());
        RefStream<Tensor> tensorStream = temp_20_0015.stream();
        temp_20_0015.freeRef();
        if (1 == leftLength) {
          tensorStream = RefStream.of(RefUtil.get(tensorStream.reduce((a, b) -> {
            return Tensor.add(a, b);
          })));
        }
        @Nonnull final TensorList array = new TensorArray(tensorStream.toArray(Tensor[]::new));
        DeltaSet<UUID> buffer1 = buffer == null ? null : buffer.addRef();
        accumulator0.accept(buffer1, array);
      }
      if (alive1) {
        RefList<Tensor> temp_20_0016 = RefIntStream.range(0, data.length()).parallel()
            .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
              @Nullable
              Tensor tensor = data.get(dataIndex);
              Tensor temp_20_0007 = diffs[dataIndex].scale(tensor.get(0) * 2.0 / diffs[dataIndex].length());
              tensor.freeRef();
              return temp_20_0007;
            }, data.addRef(), RefUtil.addRef(diffs))).collect(RefCollectors.toList());
        RefStream<Tensor> tensorStream = temp_20_0016.stream();
        temp_20_0016.freeRef();
        if (1 == rightLength) {
          tensorStream = RefStream.of(RefUtil.get(tensorStream.reduce((a, b) -> {
            return Tensor.add(a, b);
          })));
        }
        @Nonnull final TensorList array = new TensorArray(tensorStream.map(x -> {
          Tensor temp_20_0009 = x.scale(-1);
          x.freeRef();
          return temp_20_0009;
        }).toArray(Tensor[]::new));
        DeltaSet<UUID> buffer1 = buffer == null ? null : buffer.addRef();
        accumulator1.accept(buffer1, array);
      }
      data.freeRef();
      if (null != buffer)
        buffer.freeRef();
    }

    public @SuppressWarnings("unused")
    void _free() {
      super._free();
      accumulator0.freeRef();
      accumulator1.freeRef();
      RefUtil.freeRef(diffs);
    }
  }
}
