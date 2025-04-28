module     p2_gg_httbar_d87h0l131
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d87h0l131.f90
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
      use p2_gg_httbar_abbrevd87h0
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(15) :: acd87
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd87(1)=dotproduct(ninjaE3,spvae2k2)
      acd87(2)=dotproduct(ninjaE3,spval5e1)
      acd87(3)=dotproduct(ninjaE3,spvae1e2)
      acd87(4)=abb87(9)
      acd87(5)=dotproduct(ninjaE3,spval3e1)
      acd87(6)=abb87(13)
      acd87(7)=dotproduct(ninjaE3,spvae1k2)
      acd87(8)=dotproduct(ninjaE3,spval4e2)
      acd87(9)=dotproduct(ninjaE3,spvae2e1)
      acd87(10)=abb87(25)
      acd87(11)=dotproduct(ninjaE3,spvae1l3)
      acd87(12)=abb87(30)
      acd87(13)=acd87(10)*acd87(7)
      acd87(14)=-acd87(12)*acd87(11)
      acd87(13)=acd87(14)+acd87(13)
      acd87(13)=acd87(13)*acd87(9)*acd87(8)
      acd87(14)=acd87(4)*acd87(2)
      acd87(15)=acd87(6)*acd87(5)
      acd87(14)=acd87(14)+acd87(15)
      acd87(14)=acd87(14)*acd87(3)*acd87(1)
      acd87(13)=acd87(14)+acd87(13)
      brack(ninjaidxt2mu0)=acd87(13)
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine brack_32:
   pure subroutine brack_32(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd87h0
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(80) :: acd87
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd87(1)=dotproduct(ninjaE3,spvae2k2)
      acd87(2)=dotproduct(ninjaE3,spval5e1)
      acd87(3)=dotproduct(ninjaE4,spvae1e2)
      acd87(4)=abb87(9)
      acd87(5)=dotproduct(ninjaE3,spvae1e2)
      acd87(6)=dotproduct(ninjaE4,spval5e1)
      acd87(7)=dotproduct(ninjaE4,spval3e1)
      acd87(8)=abb87(13)
      acd87(9)=dotproduct(ninjaE3,spval3e1)
      acd87(10)=dotproduct(ninjaE4,spvae2k2)
      acd87(11)=dotproduct(ninjaE3,spvae1k2)
      acd87(12)=dotproduct(ninjaE3,spvae2e1)
      acd87(13)=dotproduct(ninjaE4,spval4e2)
      acd87(14)=abb87(25)
      acd87(15)=dotproduct(ninjaE3,spval4e2)
      acd87(16)=dotproduct(ninjaE4,spvae2e1)
      acd87(17)=dotproduct(ninjaE4,spvae1k2)
      acd87(18)=dotproduct(ninjaE4,spvae1l3)
      acd87(19)=abb87(30)
      acd87(20)=dotproduct(ninjaE3,spvae1l3)
      acd87(21)=dotproduct(ninjaA,spvae2k2)
      acd87(22)=dotproduct(ninjaA,spval5e1)
      acd87(23)=dotproduct(ninjaA,spvae1e2)
      acd87(24)=dotproduct(ninjaA,spvae1k2)
      acd87(25)=dotproduct(ninjaA,spvae2e1)
      acd87(26)=dotproduct(ninjaA,spval3e1)
      acd87(27)=dotproduct(ninjaA,spval4e2)
      acd87(28)=dotproduct(ninjaA,spvae1l3)
      acd87(29)=abb87(26)
      acd87(30)=abb87(42)
      acd87(31)=abb87(33)
      acd87(32)=dotproduct(ninjaE3,spval4e1)
      acd87(33)=abb87(28)
      acd87(34)=abb87(12)
      acd87(35)=abb87(23)
      acd87(36)=abb87(31)
      acd87(37)=dotproduct(ninjaA,ninjaE3)
      acd87(38)=abb87(16)
      acd87(39)=dotproduct(ninjaA,spval4e1)
      acd87(40)=abb87(8)
      acd87(41)=abb87(29)
      acd87(42)=abb87(17)
      acd87(43)=dotproduct(ninjaE3,spval5k1)
      acd87(44)=abb87(10)
      acd87(45)=abb87(15)
      acd87(46)=abb87(22)
      acd87(47)=abb87(32)
      acd87(48)=dotproduct(ninjaE3,spval3k1)
      acd87(49)=abb87(14)
      acd87(50)=abb87(18)
      acd87(51)=abb87(19)
      acd87(52)=dotproduct(ninjaE3,spvak1k2)
      acd87(53)=abb87(20)
      acd87(54)=dotproduct(ninjaE3,spvak1l3)
      acd87(55)=abb87(21)
      acd87(56)=dotproduct(ninjaE3,spval4k1)
      acd87(57)=abb87(24)
      acd87(58)=abb87(27)
      acd87(59)=dotproduct(ninjaE3,spvak2e1)
      acd87(60)=abb87(37)
      acd87(61)=acd87(19)*acd87(18)
      acd87(62)=acd87(14)*acd87(17)
      acd87(61)=acd87(61)-acd87(62)
      acd87(61)=acd87(61)*acd87(15)
      acd87(62)=acd87(19)*acd87(20)
      acd87(63)=acd87(14)*acd87(11)
      acd87(62)=acd87(62)-acd87(63)
      acd87(63)=-acd87(13)*acd87(62)
      acd87(63)=-acd87(61)+acd87(63)
      acd87(63)=acd87(12)*acd87(63)
      acd87(64)=acd87(8)*acd87(7)
      acd87(65)=acd87(4)*acd87(6)
      acd87(64)=acd87(64)+acd87(65)
      acd87(64)=acd87(64)*acd87(1)
      acd87(65)=acd87(8)*acd87(9)
      acd87(66)=acd87(4)*acd87(2)
      acd87(65)=acd87(65)+acd87(66)
      acd87(66)=acd87(10)*acd87(65)
      acd87(66)=acd87(64)+acd87(66)
      acd87(66)=acd87(5)*acd87(66)
      acd87(67)=acd87(62)*acd87(15)
      acd87(68)=-acd87(16)*acd87(67)
      acd87(69)=acd87(65)*acd87(1)
      acd87(70)=acd87(3)*acd87(69)
      acd87(63)=acd87(66)+acd87(63)+acd87(68)+acd87(70)
      acd87(62)=acd87(62)*acd87(27)
      acd87(66)=-acd87(19)*acd87(28)
      acd87(68)=acd87(14)*acd87(24)
      acd87(66)=acd87(68)+acd87(35)+acd87(66)
      acd87(66)=acd87(15)*acd87(66)
      acd87(68)=acd87(20)*acd87(36)
      acd87(70)=acd87(11)*acd87(34)
      acd87(66)=acd87(66)+acd87(68)+acd87(70)-acd87(62)
      acd87(66)=acd87(12)*acd87(66)
      acd87(65)=acd87(65)*acd87(21)
      acd87(68)=acd87(32)*acd87(33)
      acd87(65)=acd87(68)+acd87(65)
      acd87(68)=acd87(8)*acd87(26)
      acd87(70)=acd87(4)*acd87(22)
      acd87(68)=acd87(70)+acd87(29)+acd87(68)
      acd87(68)=acd87(1)*acd87(68)
      acd87(70)=acd87(9)*acd87(31)
      acd87(71)=acd87(2)*acd87(30)
      acd87(68)=acd87(68)+acd87(71)+acd87(70)+acd87(65)
      acd87(68)=acd87(5)*acd87(68)
      acd87(67)=-acd87(25)*acd87(67)
      acd87(69)=acd87(23)*acd87(69)
      acd87(66)=acd87(68)+acd87(66)+acd87(67)+acd87(69)
      acd87(67)=acd87(21)*acd87(26)
      acd87(68)=ninjaP*acd87(9)
      acd87(69)=acd87(10)*acd87(68)
      acd87(67)=acd87(67)+acd87(69)
      acd87(67)=acd87(8)*acd87(67)
      acd87(69)=acd87(21)*acd87(22)
      acd87(70)=ninjaP*acd87(2)
      acd87(71)=acd87(10)*acd87(70)
      acd87(69)=acd87(69)+acd87(71)
      acd87(69)=acd87(4)*acd87(69)
      acd87(64)=ninjaP*acd87(64)
      acd87(71)=acd87(33)*acd87(39)
      acd87(72)=acd87(26)*acd87(31)
      acd87(73)=acd87(22)*acd87(30)
      acd87(74)=acd87(21)*acd87(29)
      acd87(64)=acd87(64)+acd87(69)+acd87(67)+acd87(74)+acd87(73)+acd87(72)+acd&
      &87(42)+acd87(71)
      acd87(64)=acd87(5)*acd87(64)
      acd87(67)=-acd87(27)*acd87(28)
      acd87(69)=ninjaP*acd87(20)
      acd87(71)=-acd87(13)*acd87(69)
      acd87(67)=acd87(67)+acd87(71)
      acd87(67)=acd87(19)*acd87(67)
      acd87(71)=acd87(27)*acd87(24)
      acd87(72)=ninjaP*acd87(11)
      acd87(73)=acd87(13)*acd87(72)
      acd87(71)=acd87(71)+acd87(73)
      acd87(71)=acd87(14)*acd87(71)
      acd87(61)=-ninjaP*acd87(61)
      acd87(73)=acd87(28)*acd87(36)
      acd87(74)=acd87(24)*acd87(34)
      acd87(75)=acd87(27)*acd87(35)
      acd87(61)=acd87(61)+acd87(71)+acd87(67)+acd87(75)+acd87(74)+acd87(46)+acd&
      &87(73)
      acd87(61)=acd87(12)*acd87(61)
      acd87(67)=-acd87(25)*acd87(28)
      acd87(69)=-acd87(16)*acd87(69)
      acd87(67)=acd87(67)+acd87(69)
      acd87(67)=acd87(19)*acd87(67)
      acd87(69)=acd87(25)*acd87(24)
      acd87(71)=acd87(16)*acd87(72)
      acd87(69)=acd87(69)+acd87(71)
      acd87(69)=acd87(14)*acd87(69)
      acd87(71)=acd87(25)*acd87(35)
      acd87(67)=acd87(69)+acd87(67)+acd87(50)+acd87(71)
      acd87(67)=acd87(15)*acd87(67)
      acd87(69)=acd87(23)*acd87(26)
      acd87(68)=acd87(3)*acd87(68)
      acd87(68)=acd87(69)+acd87(68)
      acd87(68)=acd87(8)*acd87(68)
      acd87(69)=acd87(23)*acd87(22)
      acd87(70)=acd87(3)*acd87(70)
      acd87(69)=acd87(69)+acd87(70)
      acd87(69)=acd87(4)*acd87(69)
      acd87(70)=acd87(23)*acd87(29)
      acd87(68)=acd87(69)+acd87(68)+acd87(40)+acd87(70)
      acd87(68)=acd87(1)*acd87(68)
      acd87(62)=-acd87(25)*acd87(62)
      acd87(65)=acd87(23)*acd87(65)
      acd87(69)=acd87(59)*acd87(60)
      acd87(70)=acd87(56)*acd87(57)
      acd87(71)=acd87(54)*acd87(55)
      acd87(72)=acd87(52)*acd87(53)
      acd87(73)=acd87(48)*acd87(49)
      acd87(74)=acd87(43)*acd87(44)
      acd87(75)=acd87(37)*acd87(38)
      acd87(76)=acd87(32)*acd87(58)
      acd87(77)=acd87(25)*acd87(36)
      acd87(77)=acd87(51)+acd87(77)
      acd87(77)=acd87(20)*acd87(77)
      acd87(78)=acd87(25)*acd87(34)
      acd87(78)=acd87(45)+acd87(78)
      acd87(78)=acd87(11)*acd87(78)
      acd87(79)=acd87(23)*acd87(31)
      acd87(79)=acd87(47)+acd87(79)
      acd87(79)=acd87(9)*acd87(79)
      acd87(80)=acd87(23)*acd87(30)
      acd87(80)=acd87(41)+acd87(80)
      acd87(80)=acd87(2)*acd87(80)
      acd87(61)=acd87(64)+acd87(61)+acd87(68)+acd87(67)+acd87(80)+acd87(79)+acd&
      &87(78)+acd87(77)+acd87(76)+2.0_ki*acd87(75)+acd87(74)+acd87(73)+acd87(72&
      &)+acd87(71)+acd87(69)+acd87(70)+acd87(65)+acd87(62)
      brack(ninjaidxt1mu0)=acd87(66)
      brack(ninjaidxt0mu0)=acd87(61)
      brack(ninjaidxt0mu2)=acd87(63)
   end subroutine brack_32
!---#] subroutine brack_32:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d87h0_ninja_t3")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd87h0
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a, b, c
      complex(ki), intent(in) :: param
      complex(ki), dimension(4) :: vecA, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = k2-k3-k5
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
end module     p2_gg_httbar_d87h0l131
