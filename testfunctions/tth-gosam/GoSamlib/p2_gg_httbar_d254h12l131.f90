module     p2_gg_httbar_d254h12l131
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity12d254h12l131.f90
   ! generator: buildfortran_tn3.py
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_util, only: cond_t, d => metric_tensor
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   integer, parameter :: ninjaidxt2mu0 = 0
   integer, parameter :: ninjaidxt1mu0 = 1
   integer, parameter :: ninjaidxt0mu0 = 2
   integer, parameter :: ninjaidxt0mu2 = 3
   public :: numerator_t3
contains
!---#[ subroutine brack_31:
   pure subroutine brack_31(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd254h12
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(1) :: acd254
      complex(ki), dimension (0:*), intent(inout) :: brack
      brack(ninjaidxt2mu0)=0.0_ki
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine brack_32:
   pure subroutine brack_32(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd254h12
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(89) :: acd254
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd254(1)=dotproduct(ninjaE3,spvae1l5)
      acd254(2)=dotproduct(ninjaE3,spvae2e1)
      acd254(3)=abb254(7)
      acd254(4)=dotproduct(ninjaE3,spvae1l4)
      acd254(5)=abb254(28)
      acd254(6)=dotproduct(ninjaA,ninjaE3)
      acd254(7)=dotproduct(ninjaE3,spvak2e2)
      acd254(8)=abb254(42)
      acd254(9)=dotproduct(ninjaE3,spvak1l5)
      acd254(10)=dotproduct(ninjaE3,spvae1k1)
      acd254(11)=dotproduct(ninjaE3,spvak1l4)
      acd254(12)=dotproduct(ninjaE3,spval3l5)
      acd254(13)=dotproduct(ninjaE3,spvae1l3)
      acd254(14)=dotproduct(ninjaE3,spval3l4)
      acd254(15)=dotproduct(ninjaE3,spvak2l4)
      acd254(16)=dotproduct(ninjaE3,spvae1k2)
      acd254(17)=abb254(29)
      acd254(18)=dotproduct(ninjaE3,spvak2l5)
      acd254(19)=abb254(59)
      acd254(20)=dotproduct(ninjaE3,spvak2e1)
      acd254(21)=dotproduct(ninjaE3,spvae2l4)
      acd254(22)=dotproduct(ninjaE3,spvae1e2)
      acd254(23)=abb254(9)
      acd254(24)=dotproduct(ninjaA,ninjaA)
      acd254(25)=dotproduct(ninjaA,spvae1l5)
      acd254(26)=dotproduct(ninjaA,spvae2e1)
      acd254(27)=dotproduct(ninjaA,spvae1l4)
      acd254(28)=abb254(34)
      acd254(29)=abb254(35)
      acd254(30)=abb254(46)
      acd254(31)=abb254(20)
      acd254(32)=dotproduct(ninjaA,spvak1l5)
      acd254(33)=dotproduct(ninjaA,spvae1k1)
      acd254(34)=dotproduct(ninjaA,spval3l5)
      acd254(35)=dotproduct(ninjaA,spvae1l3)
      acd254(36)=dotproduct(ninjaA,spvak2e1)
      acd254(37)=dotproduct(ninjaA,spvae2l4)
      acd254(38)=dotproduct(ninjaA,spvae1e2)
      acd254(39)=dotproduct(ninjaA,spvak2e2)
      acd254(40)=dotproduct(ninjaA,spvak1l4)
      acd254(41)=dotproduct(ninjaA,spval3l4)
      acd254(42)=dotproduct(ninjaA,spvak2l4)
      acd254(43)=dotproduct(ninjaA,spvae1k2)
      acd254(44)=dotproduct(ninjaA,spvak2l5)
      acd254(45)=abb254(50)
      acd254(46)=abb254(32)
      acd254(47)=abb254(55)
      acd254(48)=abb254(52)
      acd254(49)=abb254(16)
      acd254(50)=abb254(19)
      acd254(51)=dotproduct(ninjaE3,spvak2k1)
      acd254(52)=abb254(31)
      acd254(53)=abb254(58)
      acd254(54)=abb254(37)
      acd254(55)=abb254(41)
      acd254(56)=abb254(47)
      acd254(57)=abb254(56)
      acd254(58)=abb254(38)
      acd254(59)=abb254(53)
      acd254(60)=abb254(27)
      acd254(61)=abb254(30)
      acd254(62)=abb254(15)
      acd254(63)=abb254(25)
      acd254(64)=dotproduct(ninjaE3,spvak1e1)
      acd254(65)=abb254(21)
      acd254(66)=dotproduct(ninjaE3,spval3e1)
      acd254(67)=abb254(54)
      acd254(68)=abb254(39)
      acd254(69)=abb254(49)
      acd254(70)=acd254(5)*acd254(4)
      acd254(71)=-acd254(3)*acd254(1)
      acd254(70)=acd254(70)+acd254(71)
      acd254(70)=acd254(2)*acd254(70)
      acd254(71)=acd254(13)*acd254(14)
      acd254(72)=2.0_ki*acd254(6)
      acd254(73)=acd254(72)*acd254(4)
      acd254(74)=acd254(10)*acd254(11)
      acd254(71)=acd254(73)+acd254(71)-acd254(74)
      acd254(71)=acd254(71)*acd254(5)
      acd254(73)=acd254(18)*acd254(19)
      acd254(74)=acd254(15)*acd254(17)
      acd254(73)=acd254(73)+acd254(74)
      acd254(73)=acd254(73)*acd254(16)
      acd254(74)=acd254(8)*acd254(1)*acd254(7)
      acd254(71)=acd254(71)+acd254(73)+acd254(74)
      acd254(73)=acd254(72)*acd254(1)
      acd254(74)=acd254(13)*acd254(12)
      acd254(75)=acd254(10)*acd254(9)
      acd254(73)=acd254(73)+acd254(74)-acd254(75)
      acd254(74)=-acd254(3)*acd254(73)
      acd254(74)=acd254(74)+acd254(71)
      acd254(74)=acd254(2)*acd254(74)
      acd254(75)=acd254(20)*acd254(23)
      acd254(76)=acd254(22)*acd254(21)*acd254(75)
      acd254(74)=acd254(76)+acd254(74)
      acd254(76)=-acd254(11)*acd254(33)
      acd254(77)=acd254(14)*acd254(35)
      acd254(78)=acd254(24)+ninjaP
      acd254(79)=acd254(4)*acd254(78)
      acd254(80)=acd254(13)*acd254(41)
      acd254(81)=-acd254(10)*acd254(40)
      acd254(82)=acd254(27)*acd254(72)
      acd254(76)=acd254(82)+acd254(81)+acd254(80)+acd254(79)+acd254(76)+acd254(&
      &77)
      acd254(76)=acd254(5)*acd254(76)
      acd254(77)=-acd254(12)*acd254(35)
      acd254(79)=acd254(9)*acd254(33)
      acd254(80)=-acd254(13)*acd254(34)
      acd254(81)=acd254(10)*acd254(32)
      acd254(78)=-acd254(1)*acd254(78)
      acd254(82)=-acd254(25)*acd254(72)
      acd254(77)=acd254(82)+acd254(78)+acd254(81)+acd254(80)+acd254(77)+acd254(&
      &79)
      acd254(77)=acd254(3)*acd254(77)
      acd254(78)=acd254(19)*acd254(44)
      acd254(79)=acd254(17)*acd254(42)
      acd254(78)=acd254(78)+acd254(79)
      acd254(78)=acd254(16)*acd254(78)
      acd254(79)=-acd254(51)*acd254(52)
      acd254(80)=acd254(19)*acd254(43)
      acd254(80)=acd254(55)+acd254(80)
      acd254(80)=acd254(18)*acd254(80)
      acd254(81)=acd254(17)*acd254(43)
      acd254(81)=acd254(54)+acd254(81)
      acd254(81)=acd254(15)*acd254(81)
      acd254(82)=acd254(14)*acd254(53)
      acd254(83)=acd254(12)*acd254(47)
      acd254(84)=acd254(4)*acd254(49)
      acd254(85)=acd254(13)*acd254(48)
      acd254(86)=acd254(10)*acd254(46)
      acd254(87)=acd254(8)*acd254(25)
      acd254(87)=acd254(50)+acd254(87)
      acd254(87)=acd254(7)*acd254(87)
      acd254(88)=acd254(8)*acd254(39)
      acd254(88)=acd254(45)+acd254(88)
      acd254(88)=acd254(1)*acd254(88)
      acd254(89)=acd254(28)*acd254(72)
      acd254(76)=acd254(77)+acd254(76)+acd254(89)+acd254(88)+acd254(87)+acd254(&
      &86)+acd254(85)+acd254(84)+acd254(78)+acd254(83)+acd254(82)+acd254(81)+ac&
      &d254(79)+acd254(80)
      acd254(76)=acd254(2)*acd254(76)
      acd254(77)=acd254(66)*acd254(67)
      acd254(78)=acd254(64)*acd254(65)
      acd254(79)=acd254(23)*acd254(37)
      acd254(79)=acd254(60)+acd254(79)
      acd254(79)=acd254(20)*acd254(79)
      acd254(80)=-acd254(9)*acd254(56)
      acd254(81)=acd254(23)*acd254(36)
      acd254(81)=acd254(62)+acd254(81)
      acd254(81)=acd254(21)*acd254(81)
      acd254(77)=acd254(81)+acd254(80)+acd254(79)+acd254(77)+acd254(78)
      acd254(77)=acd254(22)*acd254(77)
      acd254(78)=acd254(51)*acd254(63)
      acd254(75)=acd254(38)*acd254(75)
      acd254(79)=acd254(13)*acd254(59)
      acd254(80)=acd254(10)*acd254(58)
      acd254(75)=acd254(80)+acd254(79)+acd254(78)+acd254(75)
      acd254(75)=acd254(21)*acd254(75)
      acd254(78)=acd254(66)*acd254(69)
      acd254(79)=acd254(64)*acd254(68)
      acd254(80)=acd254(20)*acd254(61)
      acd254(81)=-acd254(9)*acd254(57)
      acd254(78)=acd254(81)+acd254(80)+acd254(78)+acd254(79)
      acd254(78)=acd254(7)*acd254(78)
      acd254(71)=acd254(26)*acd254(71)
      acd254(79)=acd254(21)*acd254(29)
      acd254(80)=acd254(22)*acd254(30)
      acd254(81)=acd254(7)*acd254(31)
      acd254(79)=acd254(81)+acd254(79)+acd254(80)
      acd254(72)=acd254(79)*acd254(72)
      acd254(73)=-acd254(3)*acd254(26)*acd254(73)
      acd254(71)=acd254(76)+acd254(73)+acd254(72)+acd254(78)+acd254(77)+acd254(&
      &75)+acd254(71)
      brack(ninjaidxt1mu0)=acd254(74)
      brack(ninjaidxt0mu0)=acd254(71)
      brack(ninjaidxt0mu2)=acd254(70)
   end subroutine brack_32
!---#] subroutine brack_32:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d254h12_ninja_t3")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd254h12
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a, b, c
      complex(ki), intent(in) :: param
      complex(ki), dimension(4) :: vecA, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = k2-k5
      vecA(1:4) = + a(0:3) - qshift(1:4)
      vecB(1:4) = + b(0:3)
      vecC(1:4) = + c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_31,vecA,vecB,vecC,param,coeffs)
      if (deg.le.(1+(-1))) return
      call cond_t(epspow.eq.t1,brack_32,vecA,vecB,vecC,param,coeffs)
   end subroutine numerator_t3
!---#] subroutine numerator_t3:
end module     p2_gg_httbar_d254h12l131
