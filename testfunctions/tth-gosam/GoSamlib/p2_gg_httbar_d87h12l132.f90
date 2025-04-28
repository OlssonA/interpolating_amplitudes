module     p2_gg_httbar_d87h12l132
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity12d87h12l132.f90
   ! generator: buildfortran_tn3.py
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_util, only: cond_t, d => metric_tensor
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
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd87h12
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(15) :: acd87
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd87(1)=dotproduct(ninjaE3,spvak2e2)
      acd87(2)=dotproduct(ninjaE3,spvae1l5)
      acd87(3)=dotproduct(ninjaE3,spvae2e1)
      acd87(4)=abb87(9)
      acd87(5)=dotproduct(ninjaE3,spvae1l3)
      acd87(6)=abb87(13)
      acd87(7)=dotproduct(ninjaE3,spvak2e1)
      acd87(8)=dotproduct(ninjaE3,spvae2l4)
      acd87(9)=dotproduct(ninjaE3,spvae1e2)
      acd87(10)=abb87(28)
      acd87(11)=dotproduct(ninjaE3,spval3e1)
      acd87(12)=abb87(39)
      acd87(13)=-acd87(10)*acd87(7)
      acd87(14)=acd87(12)*acd87(11)
      acd87(13)=acd87(14)+acd87(13)
      acd87(13)=acd87(13)*acd87(9)*acd87(8)
      acd87(14)=acd87(4)*acd87(2)
      acd87(15)=acd87(6)*acd87(5)
      acd87(14)=acd87(14)+acd87(15)
      acd87(14)=acd87(14)*acd87(3)*acd87(1)
      acd87(13)=acd87(14)+acd87(13)
      brack(ninjaidxt1x0mu0)=acd87(13)
   end subroutine brack_21
!---#] subroutine brack_21:
!---#[ subroutine brack_22:
   pure subroutine brack_22(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd87h12
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(45) :: acd87
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd87(1)=dotproduct(ninjaA1,spvak2e2)
      acd87(2)=dotproduct(ninjaE3,spvae1l5)
      acd87(3)=dotproduct(ninjaE3,spvae2e1)
      acd87(4)=abb87(9)
      acd87(5)=dotproduct(ninjaE3,spvae1l3)
      acd87(6)=abb87(13)
      acd87(7)=dotproduct(ninjaA1,spvae1l5)
      acd87(8)=dotproduct(ninjaE3,spvak2e2)
      acd87(9)=dotproduct(ninjaA1,spvae2e1)
      acd87(10)=dotproduct(ninjaA1,spvak2e1)
      acd87(11)=dotproduct(ninjaE3,spvae1e2)
      acd87(12)=dotproduct(ninjaE3,spvae2l4)
      acd87(13)=abb87(28)
      acd87(14)=dotproduct(ninjaA1,spvae1e2)
      acd87(15)=dotproduct(ninjaE3,spvak2e1)
      acd87(16)=dotproduct(ninjaE3,spval3e1)
      acd87(17)=abb87(39)
      acd87(18)=dotproduct(ninjaA1,spvae1l3)
      acd87(19)=dotproduct(ninjaA1,spvae2l4)
      acd87(20)=dotproduct(ninjaA1,spval3e1)
      acd87(21)=dotproduct(ninjaA0,spvak2e2)
      acd87(22)=dotproduct(ninjaA0,spvae1l5)
      acd87(23)=dotproduct(ninjaA0,spvae2e1)
      acd87(24)=dotproduct(ninjaA0,spvak2e1)
      acd87(25)=dotproduct(ninjaA0,spvae1e2)
      acd87(26)=dotproduct(ninjaA0,spvae1l3)
      acd87(27)=dotproduct(ninjaA0,spvae2l4)
      acd87(28)=dotproduct(ninjaA0,spval3e1)
      acd87(29)=abb87(26)
      acd87(30)=abb87(35)
      acd87(31)=abb87(31)
      acd87(32)=dotproduct(ninjaE3,spvae1l4)
      acd87(33)=abb87(22)
      acd87(34)=abb87(12)
      acd87(35)=abb87(23)
      acd87(36)=abb87(32)
      acd87(37)=acd87(17)*acd87(16)
      acd87(38)=acd87(13)*acd87(15)
      acd87(37)=acd87(37)-acd87(38)
      acd87(38)=acd87(19)*acd87(37)
      acd87(39)=acd87(17)*acd87(20)
      acd87(40)=-acd87(13)*acd87(10)
      acd87(39)=acd87(39)+acd87(40)
      acd87(39)=acd87(12)*acd87(39)
      acd87(38)=acd87(39)+acd87(38)
      acd87(38)=acd87(11)*acd87(38)
      acd87(39)=acd87(6)*acd87(5)
      acd87(40)=acd87(4)*acd87(2)
      acd87(39)=acd87(39)+acd87(40)
      acd87(40)=acd87(1)*acd87(39)
      acd87(41)=acd87(6)*acd87(18)
      acd87(42)=acd87(4)*acd87(7)
      acd87(41)=acd87(41)+acd87(42)
      acd87(41)=acd87(8)*acd87(41)
      acd87(40)=acd87(41)+acd87(40)
      acd87(40)=acd87(3)*acd87(40)
      acd87(41)=acd87(37)*acd87(12)
      acd87(42)=acd87(14)*acd87(41)
      acd87(43)=acd87(39)*acd87(8)
      acd87(44)=acd87(9)*acd87(43)
      acd87(38)=acd87(40)+acd87(38)+acd87(42)+acd87(44)
      acd87(39)=acd87(21)*acd87(39)
      acd87(40)=acd87(6)*acd87(26)
      acd87(42)=acd87(4)*acd87(22)
      acd87(40)=acd87(42)+acd87(29)+acd87(40)
      acd87(40)=acd87(8)*acd87(40)
      acd87(42)=acd87(32)*acd87(33)
      acd87(44)=acd87(5)*acd87(31)
      acd87(45)=acd87(2)*acd87(30)
      acd87(39)=acd87(40)+acd87(45)+acd87(42)+acd87(44)+acd87(39)
      acd87(39)=acd87(3)*acd87(39)
      acd87(37)=acd87(27)*acd87(37)
      acd87(40)=acd87(17)*acd87(28)
      acd87(42)=-acd87(13)*acd87(24)
      acd87(40)=acd87(42)+acd87(35)+acd87(40)
      acd87(40)=acd87(12)*acd87(40)
      acd87(42)=acd87(16)*acd87(36)
      acd87(44)=acd87(15)*acd87(34)
      acd87(37)=acd87(40)+acd87(42)+acd87(44)+acd87(37)
      acd87(37)=acd87(11)*acd87(37)
      acd87(40)=acd87(25)*acd87(41)
      acd87(41)=acd87(23)*acd87(43)
      acd87(37)=acd87(39)+acd87(37)+acd87(40)+acd87(41)
      brack(ninjaidxt0x0mu0)=acd87(37)
      brack(ninjaidxt0x1mu0)=acd87(38)
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d87h12_ninja_t2")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd87h12
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a0, a1, b, c
      complex(ki), dimension(0:2), intent(in) :: param
      complex(ki), dimension(4) :: vecA0, vecA1, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = k2-k3-k5
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
end module     p2_gg_httbar_d87h12l132
