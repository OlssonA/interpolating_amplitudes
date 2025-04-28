module     p2_gg_httbar_d37h12l132
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity12d37h12l132.f90
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
      use p2_gg_httbar_abbrevd37h12
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(60) :: acd37
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd37(1)=dotproduct(k2,ninjaE3)
      acd37(2)=abb37(19)
      acd37(3)=dotproduct(l3,ninjaE3)
      acd37(4)=abb37(28)
      acd37(5)=dotproduct(l5,ninjaE3)
      acd37(6)=abb37(33)
      acd37(7)=dotproduct(ninjaE3,spvak2l5)
      acd37(8)=abb37(15)
      acd37(9)=dotproduct(ninjaE3,spvak2l3)
      acd37(10)=abb37(16)
      acd37(11)=dotproduct(ninjaE3,spvak2k1)
      acd37(12)=abb37(20)
      acd37(13)=dotproduct(ninjaE3,spvak1l5)
      acd37(14)=abb37(21)
      acd37(15)=dotproduct(ninjaE3,spvak1l3)
      acd37(16)=abb37(22)
      acd37(17)=dotproduct(ninjaE3,spvae2l5)
      acd37(18)=abb37(24)
      acd37(19)=dotproduct(ninjaE3,spvak2e2)
      acd37(20)=abb37(25)
      acd37(21)=dotproduct(ninjaE3,spvae1l5)
      acd37(22)=abb37(26)
      acd37(23)=dotproduct(ninjaE3,spvae2l3)
      acd37(24)=abb37(30)
      acd37(25)=dotproduct(ninjaE3,spvae1l3)
      acd37(26)=abb37(34)
      acd37(27)=dotproduct(ninjaE3,spval3e2)
      acd37(28)=abb37(35)
      acd37(29)=dotproduct(ninjaE3,spvak2e1)
      acd37(30)=abb37(38)
      acd37(31)=dotproduct(ninjaE3,spval5l3)
      acd37(32)=abb37(39)
      acd37(33)=dotproduct(ninjaE3,spval3k2)
      acd37(34)=abb37(40)
      acd37(35)=dotproduct(ninjaE3,spval3e1)
      acd37(36)=abb37(41)
      acd37(37)=dotproduct(ninjaE3,spval3l5)
      acd37(38)=abb37(43)
      acd37(39)=dotproduct(ninjaE3,spval3k1)
      acd37(40)=abb37(48)
      acd37(41)=acd37(2)*acd37(1)
      acd37(42)=acd37(4)*acd37(3)
      acd37(43)=acd37(6)*acd37(5)
      acd37(44)=acd37(8)*acd37(7)
      acd37(45)=acd37(10)*acd37(9)
      acd37(46)=acd37(12)*acd37(11)
      acd37(47)=acd37(14)*acd37(13)
      acd37(48)=acd37(16)*acd37(15)
      acd37(49)=acd37(18)*acd37(17)
      acd37(50)=acd37(20)*acd37(19)
      acd37(51)=acd37(22)*acd37(21)
      acd37(52)=acd37(24)*acd37(23)
      acd37(53)=acd37(26)*acd37(25)
      acd37(54)=acd37(28)*acd37(27)
      acd37(55)=acd37(30)*acd37(29)
      acd37(56)=acd37(32)*acd37(31)
      acd37(57)=acd37(34)*acd37(33)
      acd37(58)=acd37(36)*acd37(35)
      acd37(59)=acd37(38)*acd37(37)
      acd37(60)=acd37(40)*acd37(39)
      acd37(41)=acd37(60)+acd37(59)+acd37(58)+acd37(57)+acd37(56)+acd37(55)+acd&
      &37(54)+acd37(53)+acd37(52)+acd37(51)+acd37(50)+acd37(49)+acd37(48)+acd37&
      &(47)+acd37(46)+acd37(45)+acd37(44)+acd37(43)+acd37(41)+acd37(42)
      brack(ninjaidxt2x0mu0)=0.0_ki
      brack(ninjaidxt1x0mu0)=acd37(41)
      brack(ninjaidxt1x1mu0)=0.0_ki
   end subroutine brack_21
!---#] subroutine brack_21:
!---#[ subroutine brack_22:
   pure subroutine brack_22(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd37h12
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(82) :: acd37
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd37(1)=dotproduct(k2,ninjaA1)
      acd37(2)=abb37(19)
      acd37(3)=dotproduct(l3,ninjaA1)
      acd37(4)=abb37(28)
      acd37(5)=dotproduct(l5,ninjaA1)
      acd37(6)=abb37(33)
      acd37(7)=dotproduct(ninjaA1,spvak2l5)
      acd37(8)=abb37(15)
      acd37(9)=dotproduct(ninjaA1,spvak2l3)
      acd37(10)=abb37(16)
      acd37(11)=dotproduct(ninjaA1,spvak2k1)
      acd37(12)=abb37(20)
      acd37(13)=dotproduct(ninjaA1,spvak1l5)
      acd37(14)=abb37(21)
      acd37(15)=dotproduct(ninjaA1,spvak1l3)
      acd37(16)=abb37(22)
      acd37(17)=dotproduct(ninjaA1,spvae2l5)
      acd37(18)=abb37(24)
      acd37(19)=dotproduct(ninjaA1,spvak2e2)
      acd37(20)=abb37(25)
      acd37(21)=dotproduct(ninjaA1,spvae1l5)
      acd37(22)=abb37(26)
      acd37(23)=dotproduct(ninjaA1,spvae2l3)
      acd37(24)=abb37(30)
      acd37(25)=dotproduct(ninjaA1,spvae1l3)
      acd37(26)=abb37(34)
      acd37(27)=dotproduct(ninjaA1,spval3e2)
      acd37(28)=abb37(35)
      acd37(29)=dotproduct(ninjaA1,spvak2e1)
      acd37(30)=abb37(38)
      acd37(31)=dotproduct(ninjaA1,spval5l3)
      acd37(32)=abb37(39)
      acd37(33)=dotproduct(ninjaA1,spval3k2)
      acd37(34)=abb37(40)
      acd37(35)=dotproduct(ninjaA1,spval3e1)
      acd37(36)=abb37(41)
      acd37(37)=dotproduct(ninjaA1,spval3l5)
      acd37(38)=abb37(43)
      acd37(39)=dotproduct(ninjaA1,spval3k1)
      acd37(40)=abb37(48)
      acd37(41)=dotproduct(k2,ninjaA0)
      acd37(42)=dotproduct(l3,ninjaA0)
      acd37(43)=dotproduct(l5,ninjaA0)
      acd37(44)=dotproduct(ninjaA0,spvak2l5)
      acd37(45)=dotproduct(ninjaA0,spvak2l3)
      acd37(46)=dotproduct(ninjaA0,spvak2k1)
      acd37(47)=dotproduct(ninjaA0,spvak1l5)
      acd37(48)=dotproduct(ninjaA0,spvak1l3)
      acd37(49)=dotproduct(ninjaA0,spvae2l5)
      acd37(50)=dotproduct(ninjaA0,spvak2e2)
      acd37(51)=dotproduct(ninjaA0,spvae1l5)
      acd37(52)=dotproduct(ninjaA0,spvae2l3)
      acd37(53)=dotproduct(ninjaA0,spvae1l3)
      acd37(54)=dotproduct(ninjaA0,spval3e2)
      acd37(55)=dotproduct(ninjaA0,spvak2e1)
      acd37(56)=dotproduct(ninjaA0,spval5l3)
      acd37(57)=dotproduct(ninjaA0,spval3k2)
      acd37(58)=dotproduct(ninjaA0,spval3e1)
      acd37(59)=dotproduct(ninjaA0,spval3l5)
      acd37(60)=dotproduct(ninjaA0,spval3k1)
      acd37(61)=abb37(18)
      acd37(62)=acd37(1)*acd37(2)
      acd37(63)=acd37(3)*acd37(4)
      acd37(64)=acd37(5)*acd37(6)
      acd37(65)=acd37(7)*acd37(8)
      acd37(66)=acd37(9)*acd37(10)
      acd37(67)=acd37(11)*acd37(12)
      acd37(68)=acd37(13)*acd37(14)
      acd37(69)=acd37(15)*acd37(16)
      acd37(70)=acd37(17)*acd37(18)
      acd37(71)=acd37(19)*acd37(20)
      acd37(72)=acd37(21)*acd37(22)
      acd37(73)=acd37(23)*acd37(24)
      acd37(74)=acd37(25)*acd37(26)
      acd37(75)=acd37(27)*acd37(28)
      acd37(76)=acd37(29)*acd37(30)
      acd37(77)=acd37(31)*acd37(32)
      acd37(78)=acd37(33)*acd37(34)
      acd37(79)=acd37(35)*acd37(36)
      acd37(80)=acd37(37)*acd37(38)
      acd37(81)=acd37(39)*acd37(40)
      acd37(62)=acd37(81)+acd37(80)+acd37(79)+acd37(78)+acd37(77)+acd37(76)+acd&
      &37(75)+acd37(74)+acd37(73)+acd37(72)+acd37(71)+acd37(70)+acd37(69)+acd37&
      &(68)+acd37(67)+acd37(66)+acd37(65)+acd37(64)+acd37(62)+acd37(63)
      acd37(63)=acd37(41)*acd37(2)
      acd37(64)=acd37(42)*acd37(4)
      acd37(65)=acd37(43)*acd37(6)
      acd37(66)=acd37(44)*acd37(8)
      acd37(67)=acd37(45)*acd37(10)
      acd37(68)=acd37(46)*acd37(12)
      acd37(69)=acd37(47)*acd37(14)
      acd37(70)=acd37(48)*acd37(16)
      acd37(71)=acd37(49)*acd37(18)
      acd37(72)=acd37(50)*acd37(20)
      acd37(73)=acd37(51)*acd37(22)
      acd37(74)=acd37(52)*acd37(24)
      acd37(75)=acd37(53)*acd37(26)
      acd37(76)=acd37(54)*acd37(28)
      acd37(77)=acd37(55)*acd37(30)
      acd37(78)=acd37(56)*acd37(32)
      acd37(79)=acd37(57)*acd37(34)
      acd37(80)=acd37(58)*acd37(36)
      acd37(81)=acd37(59)*acd37(38)
      acd37(82)=acd37(60)*acd37(40)
      acd37(63)=acd37(61)+acd37(82)+acd37(81)+acd37(80)+acd37(79)+acd37(78)+acd&
      &37(77)+acd37(76)+acd37(75)+acd37(74)+acd37(73)+acd37(72)+acd37(71)+acd37&
      &(70)+acd37(69)+acd37(68)+acd37(67)+acd37(66)+acd37(65)+acd37(63)+acd37(6&
      &4)
      brack(ninjaidxt0x0mu0)=acd37(63)
      brack(ninjaidxt0x0mu2)=0.0_ki
      brack(ninjaidxt0x1mu0)=acd37(62)
      brack(ninjaidxt0x2mu0)=0.0_ki
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d37h12_ninja_t2")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd37h12
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a0, a1, b, c
      complex(ki), dimension(0:2), intent(in) :: param
      complex(ki), dimension(4) :: vecA0, vecA1, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      vecA0(1:4) = + a0(0:3)
      vecA1(1:4) = + a1(0:3)
      vecB(1:4) = + b(0:3)
      vecC(1:4) = + c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_21,vecA0,vecA1,vecB,vecC,param,coeffs)
      if (deg.le.(1+(0))) return
      call cond_t(epspow.eq.t1,brack_22,vecA0,vecA1,vecB,vecC,param,coeffs)
   end subroutine numerator_t2
!---#] subroutine numerator_t2:
end module     p2_gg_httbar_d37h12l132
