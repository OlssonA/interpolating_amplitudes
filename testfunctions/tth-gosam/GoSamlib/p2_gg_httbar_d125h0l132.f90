module     p2_gg_httbar_d125h0l132
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d125h0l132.f90
   ! generator: buildfortran_tn3.py
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_util, only: cond_t, d => metric_tensor
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   integer, parameter :: ninjaidxt2x0mu0 = 0
   integer, parameter :: ninjaidxt1x0mu0 = 1
   integer, parameter :: ninjaidxt1x1mu0 = 2
   integer, parameter :: ninjaidxt0x0mu0 = 3
   integer, parameter :: ninjaidxt0x0mu2 = 4
   integer, parameter :: ninjaidxt0x1mu0 = 5
   integer, parameter :: ninjaidxt0x2mu0 = 6
   public :: numerator_t2
contains
!---#[ subroutine brack_21:
   pure subroutine brack_21(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd125h0
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(13) :: acd125
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd125(1)=dotproduct(ninjaE3,spvae1k2)
      acd125(2)=dotproduct(ninjaE3,spval5e2)
      acd125(3)=dotproduct(ninjaE3,spvae2e1)
      acd125(4)=abb125(41)
      acd125(5)=dotproduct(ninjaE3,spval4e2)
      acd125(6)=abb125(42)
      acd125(7)=dotproduct(ninjaE3,spvae2k2)
      acd125(8)=dotproduct(ninjaE3,spval5e1)
      acd125(9)=dotproduct(ninjaE3,spvae1e2)
      acd125(10)=dotproduct(ninjaE3,spval4e1)
      acd125(11)=-acd125(8)*acd125(4)
      acd125(12)=-acd125(10)*acd125(6)
      acd125(11)=acd125(12)+acd125(11)
      acd125(11)=acd125(11)*acd125(9)*acd125(7)
      acd125(12)=-acd125(2)*acd125(4)
      acd125(13)=-acd125(5)*acd125(6)
      acd125(12)=acd125(12)+acd125(13)
      acd125(12)=acd125(12)*acd125(3)*acd125(1)
      acd125(11)=acd125(12)+acd125(11)
      brack(ninjaidxt2x0mu0)=0.0_ki
      brack(ninjaidxt1x0mu0)=acd125(11)
      brack(ninjaidxt1x1mu0)=0.0_ki
   end subroutine brack_21
!---#] subroutine brack_21:
!---#[ subroutine brack_22:
   pure subroutine brack_22(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd125h0
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(72) :: acd125
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd125(1)=dotproduct(ninjaA1,spvae1k2)
      acd125(2)=dotproduct(ninjaE3,spvae2e1)
      acd125(3)=dotproduct(ninjaE3,spval5e2)
      acd125(4)=abb125(41)
      acd125(5)=dotproduct(ninjaE3,spval4e2)
      acd125(6)=abb125(42)
      acd125(7)=dotproduct(ninjaA1,spvae2e1)
      acd125(8)=dotproduct(ninjaE3,spvae1k2)
      acd125(9)=dotproduct(ninjaA1,spval5e2)
      acd125(10)=dotproduct(ninjaA1,spval5e1)
      acd125(11)=dotproduct(ninjaE3,spvae2k2)
      acd125(12)=dotproduct(ninjaE3,spvae1e2)
      acd125(13)=dotproduct(ninjaA1,spval4e2)
      acd125(14)=dotproduct(ninjaA1,spval4e1)
      acd125(15)=dotproduct(ninjaA1,spvae2k2)
      acd125(16)=dotproduct(ninjaE3,spval5e1)
      acd125(17)=dotproduct(ninjaE3,spval4e1)
      acd125(18)=dotproduct(ninjaA1,spvae1e2)
      acd125(19)=dotproduct(ninjaA0,ninjaE3)
      acd125(20)=abb125(9)
      acd125(21)=abb125(39)
      acd125(22)=abb125(47)
      acd125(23)=abb125(19)
      acd125(24)=abb125(32)
      acd125(25)=abb125(35)
      acd125(26)=dotproduct(ninjaA0,spvae1k2)
      acd125(27)=dotproduct(ninjaA0,spvae2e1)
      acd125(28)=dotproduct(ninjaA0,spval5e2)
      acd125(29)=dotproduct(ninjaA0,spval5e1)
      acd125(30)=dotproduct(ninjaA0,spval4e2)
      acd125(31)=dotproduct(ninjaA0,spval4e1)
      acd125(32)=dotproduct(ninjaA0,spvae2k2)
      acd125(33)=dotproduct(ninjaA0,spvae1e2)
      acd125(34)=abb125(10)
      acd125(35)=abb125(11)
      acd125(36)=dotproduct(ninjaE3,spvak2e1)
      acd125(37)=abb125(14)
      acd125(38)=abb125(20)
      acd125(39)=dotproduct(ninjaE3,spval3e1)
      acd125(40)=abb125(22)
      acd125(41)=abb125(43)
      acd125(42)=abb125(18)
      acd125(43)=dotproduct(ninjaE3,spvae1l3)
      acd125(44)=abb125(45)
      acd125(45)=abb125(28)
      acd125(46)=abb125(17)
      acd125(47)=abb125(46)
      acd125(48)=abb125(24)
      acd125(49)=abb125(44)
      acd125(50)=abb125(34)
      acd125(51)=abb125(50)
      acd125(52)=abb125(48)
      acd125(53)=abb125(38)
      acd125(54)=abb125(37)
      acd125(55)=acd125(3)*acd125(4)
      acd125(56)=acd125(5)*acd125(6)
      acd125(55)=acd125(55)+acd125(56)
      acd125(56)=acd125(2)*acd125(55)
      acd125(57)=-acd125(1)*acd125(56)
      acd125(55)=acd125(8)*acd125(55)
      acd125(58)=-acd125(7)*acd125(55)
      acd125(59)=acd125(16)*acd125(4)
      acd125(60)=acd125(17)*acd125(6)
      acd125(59)=acd125(59)+acd125(60)
      acd125(60)=acd125(12)*acd125(59)
      acd125(61)=-acd125(15)*acd125(60)
      acd125(59)=acd125(11)*acd125(59)
      acd125(62)=-acd125(18)*acd125(59)
      acd125(63)=acd125(2)*acd125(8)
      acd125(64)=acd125(63)*acd125(4)
      acd125(65)=-acd125(9)*acd125(64)
      acd125(66)=acd125(12)*acd125(11)
      acd125(67)=acd125(66)*acd125(4)
      acd125(68)=-acd125(10)*acd125(67)
      acd125(69)=acd125(63)*acd125(6)
      acd125(70)=-acd125(13)*acd125(69)
      acd125(71)=acd125(66)*acd125(6)
      acd125(72)=-acd125(14)*acd125(71)
      acd125(57)=acd125(72)+acd125(70)+acd125(68)+acd125(65)+acd125(62)+acd125(&
      &61)+acd125(57)+acd125(58)
      acd125(58)=acd125(20)*acd125(8)
      acd125(61)=acd125(21)*acd125(3)
      acd125(62)=acd125(22)*acd125(16)
      acd125(65)=acd125(23)*acd125(5)
      acd125(68)=acd125(24)*acd125(17)
      acd125(70)=acd125(25)*acd125(11)
      acd125(58)=acd125(70)+acd125(68)+acd125(65)+acd125(62)+acd125(61)+acd125(&
      &58)
      acd125(61)=2.0_ki*acd125(19)
      acd125(58)=acd125(61)*acd125(58)
      acd125(61)=acd125(35)*acd125(3)
      acd125(62)=acd125(37)*acd125(36)
      acd125(65)=acd125(38)*acd125(5)
      acd125(68)=acd125(40)*acd125(39)
      acd125(61)=acd125(68)+acd125(65)+acd125(62)+acd125(61)
      acd125(61)=acd125(8)*acd125(61)
      acd125(62)=acd125(45)*acd125(36)
      acd125(65)=acd125(49)*acd125(16)
      acd125(68)=acd125(51)*acd125(39)
      acd125(70)=acd125(52)*acd125(17)
      acd125(62)=acd125(70)+acd125(68)+acd125(65)+acd125(62)
      acd125(62)=acd125(11)*acd125(62)
      acd125(65)=acd125(44)*acd125(3)
      acd125(68)=-acd125(46)*acd125(16)
      acd125(70)=acd125(47)*acd125(5)
      acd125(72)=-acd125(48)*acd125(17)
      acd125(65)=acd125(72)+acd125(70)+acd125(68)+acd125(65)
      acd125(65)=acd125(43)*acd125(65)
      acd125(68)=acd125(41)*acd125(3)
      acd125(70)=acd125(42)*acd125(5)
      acd125(68)=acd125(70)+acd125(68)
      acd125(68)=acd125(2)*acd125(68)
      acd125(70)=acd125(50)*acd125(16)
      acd125(72)=acd125(53)*acd125(17)
      acd125(70)=acd125(72)+acd125(70)
      acd125(70)=acd125(12)*acd125(70)
      acd125(56)=-acd125(26)*acd125(56)
      acd125(55)=-acd125(27)*acd125(55)
      acd125(60)=-acd125(32)*acd125(60)
      acd125(59)=-acd125(33)*acd125(59)
      acd125(64)=-acd125(28)*acd125(64)
      acd125(67)=-acd125(29)*acd125(67)
      acd125(69)=-acd125(30)*acd125(69)
      acd125(71)=-acd125(31)*acd125(71)
      acd125(63)=acd125(34)*acd125(63)
      acd125(66)=acd125(54)*acd125(66)
      acd125(55)=acd125(66)+acd125(63)+acd125(71)+acd125(69)+acd125(67)+acd125(&
      &64)+acd125(59)+acd125(60)+acd125(56)+acd125(55)+acd125(58)+acd125(65)+ac&
      &d125(62)+acd125(61)+acd125(70)+acd125(68)
      brack(ninjaidxt0x0mu0)=acd125(55)
      brack(ninjaidxt0x0mu2)=0.0_ki
      brack(ninjaidxt0x1mu0)=acd125(57)
      brack(ninjaidxt0x2mu0)=0.0_ki
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d125h0_ninja_t2")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd125h0
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a0, a1, b, c
      complex(ki), dimension(0:2), intent(in) :: param
      complex(ki), dimension(4) :: vecA0, vecA1, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = k3-k2+k5
      vecA0(1:4) = - a0(0:3) - qshift(1:4)
      vecA1(1:4) = - a1(0:3)
      vecB(1:4) = - b(0:3)
      vecC(1:4) = - c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_21,vecA0,vecA1,vecB,vecC,param,coeffs)
      if (deg.le.(1+(0))) return
      call cond_t(epspow.eq.t1,brack_22,vecA0,vecA1,vecB,vecC,param,coeffs)
   end subroutine numerator_t2
!---#] subroutine numerator_t2:
end module     p2_gg_httbar_d125h0l132
