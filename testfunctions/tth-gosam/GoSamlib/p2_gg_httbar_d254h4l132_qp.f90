module     p2_gg_httbar_d254h4l132_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity4d254h4l132_qp.f90
   ! generator: buildfortran_tn3.py
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_util_qp, only: cond_t, d => metric_tensor
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   integer, parameter :: ninjaidxt1x0mu0 = 0
   integer, parameter :: ninjaidxt0x0mu0 = 1
   integer, parameter :: ninjaidxt0x1mu0 = 2
   public :: numerator_t2
contains
!---#[ subroutine brack_21:
   pure subroutine brack_21(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd254h4_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(1) :: acd254
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      brack(ninjaidxt1x0mu0)=0.0_ki
   end subroutine brack_21
!---#] subroutine brack_21:
!---#[ subroutine brack_22:
   pure subroutine brack_22(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd254h4_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(40) :: acd254
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd254(1)=dotproduct(k2,ninjaE3)
      acd254(2)=dotproduct(ninjaE3,spvak2e1)
      acd254(3)=dotproduct(ninjaE3,spvae1e2)
      acd254(4)=abb254(31)
      acd254(5)=dotproduct(ninjaE3,spvae1k2)
      acd254(6)=dotproduct(ninjaE3,spvae2e1)
      acd254(7)=abb254(48)
      acd254(8)=dotproduct(ninjaA0,ninjaE3)
      acd254(9)=abb254(7)
      acd254(10)=dotproduct(ninjaE3,spval5e1)
      acd254(11)=abb254(58)
      acd254(12)=abb254(11)
      acd254(13)=dotproduct(ninjaE3,spvae1l4)
      acd254(14)=abb254(37)
      acd254(15)=dotproduct(ninjaE3,spval5k2)
      acd254(16)=abb254(73)
      acd254(17)=dotproduct(ninjaE3,spvak2k1)
      acd254(18)=dotproduct(ninjaE3,spvak1e1)
      acd254(19)=dotproduct(ninjaE3,spval5k1)
      acd254(20)=dotproduct(ninjaE3,spvak2l3)
      acd254(21)=dotproduct(ninjaE3,spval3e1)
      acd254(22)=dotproduct(ninjaE3,spval5l3)
      acd254(23)=dotproduct(ninjaE3,spvae2l4)
      acd254(24)=abb254(21)
      acd254(25)=dotproduct(ninjaE3,spvak2e2)
      acd254(26)=abb254(9)
      acd254(27)=dotproduct(ninjaE3,spvak2l4)
      acd254(28)=abb254(14)
      acd254(29)=dotproduct(ninjaE3,spvak1k2)
      acd254(30)=dotproduct(ninjaE3,spvae1k1)
      acd254(31)=dotproduct(ninjaE3,spvak1l4)
      acd254(32)=dotproduct(ninjaE3,spval3k2)
      acd254(33)=dotproduct(ninjaE3,spvae1l3)
      acd254(34)=dotproduct(ninjaE3,spval3l4)
      acd254(35)=-acd254(21)*acd254(22)
      acd254(36)=acd254(18)*acd254(19)
      acd254(35)=acd254(35)+acd254(36)
      acd254(35)=acd254(11)*acd254(35)
      acd254(36)=acd254(21)*acd254(20)
      acd254(37)=-acd254(18)*acd254(17)
      acd254(36)=acd254(36)+acd254(37)
      acd254(36)=acd254(9)*acd254(36)
      acd254(37)=acd254(15)*acd254(16)
      acd254(38)=acd254(1)*acd254(4)
      acd254(37)=acd254(37)+acd254(38)
      acd254(37)=acd254(2)*acd254(37)
      acd254(38)=-acd254(11)*acd254(10)
      acd254(39)=acd254(2)*acd254(9)
      acd254(38)=acd254(38)+acd254(39)
      acd254(39)=2.0_ki*acd254(8)
      acd254(38)=acd254(38)*acd254(39)
      acd254(40)=acd254(10)*acd254(23)*acd254(24)
      acd254(35)=acd254(38)+acd254(37)+acd254(36)+acd254(40)+acd254(35)
      acd254(35)=acd254(3)*acd254(35)
      acd254(36)=acd254(27)*acd254(28)
      acd254(37)=acd254(25)*acd254(26)
      acd254(38)=acd254(1)*acd254(7)
      acd254(40)=-acd254(12)*acd254(39)
      acd254(36)=acd254(40)+acd254(38)+acd254(36)+acd254(37)
      acd254(36)=acd254(5)*acd254(36)
      acd254(37)=acd254(33)*acd254(34)
      acd254(38)=-acd254(30)*acd254(31)
      acd254(39)=acd254(13)*acd254(39)
      acd254(37)=acd254(39)+acd254(37)+acd254(38)
      acd254(37)=acd254(14)*acd254(37)
      acd254(38)=-acd254(33)*acd254(32)
      acd254(39)=acd254(30)*acd254(29)
      acd254(38)=acd254(38)+acd254(39)
      acd254(38)=acd254(12)*acd254(38)
      acd254(36)=acd254(36)+acd254(38)+acd254(37)
      acd254(36)=acd254(6)*acd254(36)
      acd254(35)=acd254(36)+acd254(35)
      brack(ninjaidxt0x0mu0)=acd254(35)
      brack(ninjaidxt0x1mu0)=0.0_ki
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d254h4_qp_ninja_t2")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd254h4_qp
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a0, a1, b, c
      complex(ki), dimension(0:2), intent(in) :: param
      complex(ki), dimension(4) :: vecA0, vecA1, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = k2-k5
      vecA0(1:4) = + a0(0:3) - qshift(1:4)
      vecA1(1:4) = + a1(0:3)
      vecB(1:4) = + b(0:3)
      vecC(1:4) = + c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_21,vecA0,vecA1,vecB,vecC,param,coeffs)
      if (deg.le.(1+(-1))) return
      call cond_t(epspow.eq.t1,brack_22,vecA0,vecA1,vecB,vecC,param,coeffs)
   end subroutine numerator_t2
!---#] subroutine numerator_t2:
end module     p2_gg_httbar_d254h4l132_qp
