module     p2_gg_httbar_d101h12l132
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity12d101h12l132.f90
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
      use p2_gg_httbar_abbrevd101h12
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(13) :: acd101
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd101(1)=dotproduct(ninjaE3,spvak2e1)
      acd101(2)=dotproduct(ninjaE3,spvae2l5)
      acd101(3)=dotproduct(ninjaE3,spvae1e2)
      acd101(4)=abb101(45)
      acd101(5)=dotproduct(ninjaE3,spvae2l4)
      acd101(6)=abb101(46)
      acd101(7)=dotproduct(ninjaE3,spvak2e2)
      acd101(8)=dotproduct(ninjaE3,spvae1l5)
      acd101(9)=dotproduct(ninjaE3,spvae2e1)
      acd101(10)=dotproduct(ninjaE3,spvae1l4)
      acd101(11)=-acd101(8)*acd101(4)
      acd101(12)=-acd101(10)*acd101(6)
      acd101(11)=acd101(12)+acd101(11)
      acd101(11)=acd101(11)*acd101(9)*acd101(7)
      acd101(12)=-acd101(2)*acd101(4)
      acd101(13)=-acd101(5)*acd101(6)
      acd101(12)=acd101(12)+acd101(13)
      acd101(12)=acd101(12)*acd101(3)*acd101(1)
      acd101(11)=acd101(12)+acd101(11)
      brack(ninjaidxt2x0mu0)=0.0_ki
      brack(ninjaidxt1x0mu0)=acd101(11)
      brack(ninjaidxt1x1mu0)=0.0_ki
   end subroutine brack_21
!---#] subroutine brack_21:
!---#[ subroutine brack_22:
   pure subroutine brack_22(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd101h12
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(80) :: acd101
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd101(1)=dotproduct(ninjaA1,spvak2e2)
      acd101(2)=dotproduct(ninjaE3,spvae2e1)
      acd101(3)=dotproduct(ninjaE3,spvae1l5)
      acd101(4)=abb101(45)
      acd101(5)=dotproduct(ninjaE3,spvae1l4)
      acd101(6)=abb101(46)
      acd101(7)=dotproduct(ninjaA1,spvak2e1)
      acd101(8)=dotproduct(ninjaE3,spvae2l4)
      acd101(9)=dotproduct(ninjaE3,spvae1e2)
      acd101(10)=dotproduct(ninjaE3,spvae2l5)
      acd101(11)=dotproduct(ninjaA1,spvae2l4)
      acd101(12)=dotproduct(ninjaE3,spvak2e1)
      acd101(13)=dotproduct(ninjaA1,spvae2l5)
      acd101(14)=dotproduct(ninjaA1,spvae1e2)
      acd101(15)=dotproduct(ninjaA1,spvae2e1)
      acd101(16)=dotproduct(ninjaE3,spvak2e2)
      acd101(17)=dotproduct(ninjaA1,spvae1l5)
      acd101(18)=dotproduct(ninjaA1,spvae1l4)
      acd101(19)=dotproduct(ninjaA0,ninjaE3)
      acd101(20)=abb101(24)
      acd101(21)=abb101(18)
      acd101(22)=abb101(40)
      acd101(23)=abb101(36)
      acd101(24)=abb101(30)
      acd101(25)=dotproduct(ninjaA0,spvak2e2)
      acd101(26)=dotproduct(ninjaA0,spvak2e1)
      acd101(27)=dotproduct(ninjaA0,spvae2l4)
      acd101(28)=dotproduct(ninjaA0,spvae2l5)
      acd101(29)=dotproduct(ninjaA0,spvae1e2)
      acd101(30)=dotproduct(ninjaA0,spvae2e1)
      acd101(31)=dotproduct(ninjaA0,spvae1l5)
      acd101(32)=dotproduct(ninjaA0,spvae1l4)
      acd101(33)=dotproduct(ninjaE3,spvak1l4)
      acd101(34)=abb101(7)
      acd101(35)=abb101(38)
      acd101(36)=dotproduct(ninjaE3,spvak1l5)
      acd101(37)=abb101(8)
      acd101(38)=abb101(16)
      acd101(39)=abb101(43)
      acd101(40)=dotproduct(ninjaE3,spval3e1)
      acd101(41)=abb101(33)
      acd101(42)=dotproduct(ninjaE3,spvak1e1)
      acd101(43)=abb101(39)
      acd101(44)=abb101(49)
      acd101(45)=abb101(34)
      acd101(46)=abb101(23)
      acd101(47)=dotproduct(ninjaE3,spvae1l3)
      acd101(48)=abb101(31)
      acd101(49)=dotproduct(ninjaE3,spvae1k1)
      acd101(50)=abb101(47)
      acd101(51)=dotproduct(ninjaE3,spvak2k1)
      acd101(52)=abb101(44)
      acd101(53)=abb101(21)
      acd101(54)=abb101(41)
      acd101(55)=abb101(48)
      acd101(56)=abb101(37)
      acd101(57)=abb101(42)
      acd101(58)=abb101(35)
      acd101(59)=abb101(22)
      acd101(60)=abb101(27)
      acd101(61)=abb101(29)
      acd101(62)=acd101(3)*acd101(4)
      acd101(63)=acd101(5)*acd101(6)
      acd101(62)=acd101(62)+acd101(63)
      acd101(63)=acd101(2)*acd101(62)
      acd101(64)=-acd101(1)*acd101(63)
      acd101(65)=acd101(8)*acd101(6)
      acd101(66)=acd101(10)*acd101(4)
      acd101(65)=acd101(65)+acd101(66)
      acd101(66)=acd101(9)*acd101(65)
      acd101(67)=-acd101(7)*acd101(66)
      acd101(65)=acd101(12)*acd101(65)
      acd101(68)=-acd101(14)*acd101(65)
      acd101(62)=acd101(16)*acd101(62)
      acd101(69)=-acd101(15)*acd101(62)
      acd101(70)=acd101(12)*acd101(9)
      acd101(71)=acd101(70)*acd101(6)
      acd101(72)=-acd101(11)*acd101(71)
      acd101(73)=acd101(70)*acd101(4)
      acd101(74)=-acd101(13)*acd101(73)
      acd101(75)=acd101(2)*acd101(16)
      acd101(76)=acd101(75)*acd101(4)
      acd101(77)=-acd101(17)*acd101(76)
      acd101(78)=acd101(75)*acd101(6)
      acd101(79)=-acd101(18)*acd101(78)
      acd101(64)=acd101(79)+acd101(77)+acd101(74)+acd101(72)+acd101(69)+acd101(&
      &68)+acd101(64)+acd101(67)
      acd101(67)=2.0_ki*acd101(19)
      acd101(68)=acd101(23)*acd101(67)
      acd101(69)=acd101(35)*acd101(33)
      acd101(72)=acd101(44)*acd101(36)
      acd101(74)=acd101(46)*acd101(8)
      acd101(77)=acd101(53)*acd101(10)
      acd101(79)=acd101(57)*acd101(40)
      acd101(80)=acd101(58)*acd101(42)
      acd101(68)=acd101(80)+acd101(79)+acd101(77)+acd101(74)+acd101(72)+acd101(&
      &69)+acd101(68)
      acd101(68)=acd101(9)*acd101(68)
      acd101(69)=acd101(20)*acd101(67)
      acd101(72)=acd101(34)*acd101(33)
      acd101(74)=acd101(37)*acd101(36)
      acd101(77)=acd101(38)*acd101(12)
      acd101(79)=acd101(41)*acd101(40)
      acd101(80)=acd101(43)*acd101(42)
      acd101(69)=acd101(80)+acd101(79)+acd101(77)+acd101(74)+acd101(72)+acd101(&
      &69)
      acd101(69)=acd101(16)*acd101(69)
      acd101(72)=acd101(24)*acd101(67)
      acd101(74)=acd101(59)*acd101(47)
      acd101(77)=acd101(60)*acd101(49)
      acd101(79)=acd101(61)*acd101(51)
      acd101(72)=acd101(79)+acd101(77)+acd101(74)+acd101(72)
      acd101(72)=acd101(2)*acd101(72)
      acd101(74)=acd101(21)*acd101(67)
      acd101(77)=acd101(48)*acd101(47)
      acd101(79)=acd101(50)*acd101(49)
      acd101(80)=acd101(52)*acd101(51)
      acd101(74)=acd101(80)+acd101(79)+acd101(77)+acd101(74)
      acd101(74)=acd101(8)*acd101(74)
      acd101(67)=acd101(22)*acd101(67)
      acd101(77)=acd101(54)*acd101(47)
      acd101(79)=acd101(55)*acd101(49)
      acd101(80)=acd101(56)*acd101(51)
      acd101(67)=acd101(80)+acd101(79)+acd101(77)+acd101(67)
      acd101(67)=acd101(10)*acd101(67)
      acd101(63)=-acd101(25)*acd101(63)
      acd101(66)=-acd101(26)*acd101(66)
      acd101(65)=-acd101(29)*acd101(65)
      acd101(62)=-acd101(30)*acd101(62)
      acd101(71)=-acd101(27)*acd101(71)
      acd101(73)=-acd101(28)*acd101(73)
      acd101(76)=-acd101(31)*acd101(76)
      acd101(77)=-acd101(32)*acd101(78)
      acd101(75)=acd101(39)*acd101(75)
      acd101(70)=-acd101(45)*acd101(70)
      acd101(62)=acd101(70)+acd101(75)+acd101(77)+acd101(76)+acd101(73)+acd101(&
      &71)+acd101(62)+acd101(65)+acd101(63)+acd101(66)+acd101(68)+acd101(69)+ac&
      &d101(67)+acd101(74)+acd101(72)
      brack(ninjaidxt0x0mu0)=acd101(62)
      brack(ninjaidxt0x0mu2)=0.0_ki
      brack(ninjaidxt0x1mu0)=acd101(64)
      brack(ninjaidxt0x2mu0)=0.0_ki
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d101h12_ninja_t2")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd101h12
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a0, a1, b, c
      complex(ki), dimension(0:2), intent(in) :: param
      complex(ki), dimension(4) :: vecA0, vecA1, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = k2-k4
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
end module     p2_gg_httbar_d101h12l132
