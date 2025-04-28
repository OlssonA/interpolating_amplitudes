module     p2_gg_httbar_d81h12l132
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity12d81h12l132.f90
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
      use p2_gg_httbar_abbrevd81h12
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(15) :: acd81
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd81(1)=dotproduct(ninjaE3,spvak2e2)
      acd81(2)=dotproduct(ninjaE3,spvae1l4)
      acd81(3)=dotproduct(ninjaE3,spvae2e1)
      acd81(4)=abb81(9)
      acd81(5)=dotproduct(ninjaE3,spvae1l3)
      acd81(6)=abb81(33)
      acd81(7)=dotproduct(ninjaE3,spvak2e1)
      acd81(8)=dotproduct(ninjaE3,spvae2l5)
      acd81(9)=dotproduct(ninjaE3,spvae1e2)
      acd81(10)=abb81(10)
      acd81(11)=dotproduct(ninjaE3,spval3e1)
      acd81(12)=abb81(38)
      acd81(13)=acd81(10)*acd81(7)
      acd81(14)=-acd81(12)*acd81(11)
      acd81(13)=acd81(14)+acd81(13)
      acd81(13)=acd81(13)*acd81(9)*acd81(8)
      acd81(14)=acd81(4)*acd81(2)
      acd81(15)=acd81(6)*acd81(5)
      acd81(14)=acd81(14)+acd81(15)
      acd81(14)=acd81(14)*acd81(3)*acd81(1)
      acd81(13)=acd81(14)+acd81(13)
      brack(ninjaidxt1x0mu0)=acd81(13)
   end subroutine brack_21
!---#] subroutine brack_21:
!---#[ subroutine brack_22:
   pure subroutine brack_22(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd81h12
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(48) :: acd81
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd81(1)=dotproduct(ninjaA1,spvak2e2)
      acd81(2)=dotproduct(ninjaE3,spvae1l4)
      acd81(3)=dotproduct(ninjaE3,spvae2e1)
      acd81(4)=abb81(9)
      acd81(5)=dotproduct(ninjaE3,spvae1l3)
      acd81(6)=abb81(33)
      acd81(7)=dotproduct(ninjaA1,spvae1l4)
      acd81(8)=dotproduct(ninjaE3,spvak2e2)
      acd81(9)=dotproduct(ninjaA1,spvae2e1)
      acd81(10)=dotproduct(ninjaA1,spvak2e1)
      acd81(11)=dotproduct(ninjaE3,spvae2l5)
      acd81(12)=dotproduct(ninjaE3,spvae1e2)
      acd81(13)=abb81(10)
      acd81(14)=dotproduct(ninjaA1,spvae2l5)
      acd81(15)=dotproduct(ninjaE3,spvak2e1)
      acd81(16)=dotproduct(ninjaE3,spval3e1)
      acd81(17)=abb81(38)
      acd81(18)=dotproduct(ninjaA1,spvae1e2)
      acd81(19)=dotproduct(ninjaA1,spvae1l3)
      acd81(20)=dotproduct(ninjaA1,spval3e1)
      acd81(21)=dotproduct(ninjaA0,spvak2e2)
      acd81(22)=dotproduct(ninjaA0,spvae1l4)
      acd81(23)=dotproduct(ninjaA0,spvae2e1)
      acd81(24)=dotproduct(ninjaA0,spvak2e1)
      acd81(25)=dotproduct(ninjaA0,spvae2l5)
      acd81(26)=dotproduct(ninjaA0,spvae1e2)
      acd81(27)=dotproduct(ninjaA0,spvae1l3)
      acd81(28)=dotproduct(ninjaA0,spval3e1)
      acd81(29)=abb81(23)
      acd81(30)=dotproduct(ninjaE3,spval3e2)
      acd81(31)=abb81(43)
      acd81(32)=abb81(16)
      acd81(33)=abb81(18)
      acd81(34)=dotproduct(ninjaE3,spvae2l4)
      acd81(35)=abb81(25)
      acd81(36)=abb81(45)
      acd81(37)=dotproduct(ninjaE3,spvae2l3)
      acd81(38)=abb81(39)
      acd81(39)=acd81(6)*acd81(5)
      acd81(40)=acd81(4)*acd81(2)
      acd81(39)=acd81(39)+acd81(40)
      acd81(40)=acd81(1)*acd81(39)
      acd81(41)=acd81(6)*acd81(19)
      acd81(42)=acd81(4)*acd81(7)
      acd81(41)=acd81(41)+acd81(42)
      acd81(41)=acd81(8)*acd81(41)
      acd81(40)=acd81(41)+acd81(40)
      acd81(40)=acd81(3)*acd81(40)
      acd81(41)=acd81(17)*acd81(16)
      acd81(42)=acd81(13)*acd81(15)
      acd81(41)=acd81(41)-acd81(42)
      acd81(42)=-acd81(14)*acd81(41)
      acd81(43)=-acd81(17)*acd81(20)
      acd81(44)=acd81(13)*acd81(10)
      acd81(43)=acd81(43)+acd81(44)
      acd81(43)=acd81(11)*acd81(43)
      acd81(42)=acd81(43)+acd81(42)
      acd81(42)=acd81(12)*acd81(42)
      acd81(43)=acd81(41)*acd81(11)
      acd81(44)=-acd81(18)*acd81(43)
      acd81(45)=acd81(39)*acd81(8)
      acd81(46)=acd81(9)*acd81(45)
      acd81(40)=acd81(42)+acd81(40)+acd81(44)+acd81(46)
      acd81(41)=-acd81(25)*acd81(41)
      acd81(42)=-acd81(17)*acd81(28)
      acd81(44)=acd81(13)*acd81(24)
      acd81(42)=acd81(44)+acd81(33)+acd81(42)
      acd81(42)=acd81(11)*acd81(42)
      acd81(44)=acd81(37)*acd81(38)
      acd81(46)=acd81(34)*acd81(35)
      acd81(47)=acd81(16)*acd81(36)
      acd81(48)=acd81(15)*acd81(32)
      acd81(41)=acd81(42)+acd81(48)+acd81(47)+acd81(44)+acd81(46)+acd81(41)
      acd81(41)=acd81(12)*acd81(41)
      acd81(39)=acd81(21)*acd81(39)
      acd81(42)=acd81(6)*acd81(27)
      acd81(44)=acd81(4)*acd81(22)
      acd81(42)=acd81(44)+acd81(29)+acd81(42)
      acd81(42)=acd81(8)*acd81(42)
      acd81(44)=acd81(30)*acd81(31)
      acd81(39)=acd81(42)+acd81(44)+acd81(39)
      acd81(39)=acd81(3)*acd81(39)
      acd81(42)=-acd81(26)*acd81(43)
      acd81(43)=acd81(23)*acd81(45)
      acd81(39)=acd81(41)+acd81(39)+acd81(42)+acd81(43)
      brack(ninjaidxt0x0mu0)=acd81(39)
      brack(ninjaidxt0x1mu0)=acd81(40)
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d81h12_ninja_t2")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd81h12
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a0, a1, b, c
      complex(ki), dimension(0:2), intent(in) :: param
      complex(ki), dimension(4) :: vecA0, vecA1, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = k2-k3-k4
      vecA0(1:4) = - a0(0:3) - qshift(1:4)
      vecA1(1:4) = - a1(0:3)
      vecB(1:4) = - b(0:3)
      vecC(1:4) = - c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_21,vecA0,vecA1,vecB,vecC,param,coeffs)
      if (deg.le.(1+(-1))) return
      call cond_t(epspow.eq.t1,brack_22,vecA0,vecA1,vecB,vecC,param,coeffs)
   end subroutine numerator_t2
!---#] subroutine numerator_t2:
end module     p2_gg_httbar_d81h12l132
