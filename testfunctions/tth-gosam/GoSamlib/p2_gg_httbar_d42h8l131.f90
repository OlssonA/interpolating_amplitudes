module     p2_gg_httbar_d42h8l131
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity8d42h8l131.f90
   ! generator: buildfortran_tn3.py
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_util, only: cond_t, d => metric_tensor
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   integer, parameter :: ninjaidxt3mu0 = 0
   integer, parameter :: ninjaidxt2mu0 = 1
   integer, parameter :: ninjaidxt1mu0 = 2
   integer, parameter :: ninjaidxt1mu2 = 3
   integer, parameter :: ninjaidxt0mu0 = 4
   integer, parameter :: ninjaidxt0mu2 = 5
   public :: numerator_t3
contains
!---#[ subroutine brack_31:
   pure subroutine brack_31(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd42h8
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(87) :: acd42
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd42(1)=dotproduct(k1,ninjaE3)
      acd42(2)=abb42(23)
      acd42(3)=dotproduct(k2,ninjaE3)
      acd42(4)=abb42(21)
      acd42(5)=dotproduct(l5,ninjaE3)
      acd42(6)=abb42(36)
      acd42(7)=dotproduct(ninjaE3,spvak2k1)
      acd42(8)=abb42(15)
      acd42(9)=dotproduct(ninjaE3,spvak1l4)
      acd42(10)=abb42(16)
      acd42(11)=dotproduct(ninjaE3,spvak1l5)
      acd42(12)=abb42(18)
      acd42(13)=dotproduct(ninjaE3,spvak1k2)
      acd42(14)=abb42(19)
      acd42(15)=dotproduct(ninjaE3,spvae2k1)
      acd42(16)=abb42(22)
      acd42(17)=dotproduct(ninjaE3,spvae1l5)
      acd42(18)=abb42(24)
      acd42(19)=dotproduct(ninjaE3,spvae1e2)
      acd42(20)=abb42(26)
      acd42(21)=dotproduct(ninjaE3,spvae2e1)
      acd42(22)=abb42(28)
      acd42(23)=dotproduct(ninjaE3,spvak1e1)
      acd42(24)=abb42(35)
      acd42(25)=dotproduct(ninjaE3,spval5e2)
      acd42(26)=abb42(37)
      acd42(27)=dotproduct(ninjaE3,spvae2k2)
      acd42(28)=abb42(40)
      acd42(29)=dotproduct(ninjaE3,spvak2l5)
      acd42(30)=abb42(42)
      acd42(31)=dotproduct(ninjaE3,spval5k1)
      acd42(32)=abb42(43)
      acd42(33)=dotproduct(ninjaE3,spval4k1)
      acd42(34)=abb42(46)
      acd42(35)=dotproduct(ninjaE3,spvae2l5)
      acd42(36)=abb42(47)
      acd42(37)=dotproduct(ninjaE3,spval5e1)
      acd42(38)=abb42(48)
      acd42(39)=dotproduct(ninjaE3,spvae1k1)
      acd42(40)=abb42(50)
      acd42(41)=dotproduct(ninjaE3,spval5l4)
      acd42(42)=abb42(51)
      acd42(43)=dotproduct(ninjaE3,spval4k2)
      acd42(44)=abb42(52)
      acd42(45)=dotproduct(ninjaE3,spval5k2)
      acd42(46)=abb42(53)
      acd42(47)=dotproduct(ninjaE3,spvak1e2)
      acd42(48)=abb42(55)
      acd42(49)=dotproduct(ninjaE3,spvae1l4)
      acd42(50)=abb42(56)
      acd42(51)=dotproduct(ninjaE3,spval4l5)
      acd42(52)=abb42(57)
      acd42(53)=dotproduct(ninjaE3,spval4e1)
      acd42(54)=abb42(62)
      acd42(55)=dotproduct(ninjaE3,spvae1k2)
      acd42(56)=abb42(64)
      acd42(57)=dotproduct(ninjaE3,spvak2e1)
      acd42(58)=abb42(65)
      acd42(59)=acd42(2)*acd42(1)
      acd42(60)=acd42(4)*acd42(3)
      acd42(61)=acd42(6)*acd42(5)
      acd42(62)=acd42(8)*acd42(7)
      acd42(63)=acd42(10)*acd42(9)
      acd42(64)=acd42(12)*acd42(11)
      acd42(65)=acd42(14)*acd42(13)
      acd42(66)=acd42(16)*acd42(15)
      acd42(67)=acd42(18)*acd42(17)
      acd42(68)=acd42(20)*acd42(19)
      acd42(69)=acd42(22)*acd42(21)
      acd42(70)=acd42(24)*acd42(23)
      acd42(71)=acd42(26)*acd42(25)
      acd42(72)=acd42(28)*acd42(27)
      acd42(73)=acd42(30)*acd42(29)
      acd42(74)=acd42(32)*acd42(31)
      acd42(75)=acd42(34)*acd42(33)
      acd42(76)=acd42(36)*acd42(35)
      acd42(77)=acd42(38)*acd42(37)
      acd42(78)=acd42(40)*acd42(39)
      acd42(79)=acd42(42)*acd42(41)
      acd42(80)=acd42(44)*acd42(43)
      acd42(81)=acd42(46)*acd42(45)
      acd42(82)=acd42(48)*acd42(47)
      acd42(83)=acd42(50)*acd42(49)
      acd42(84)=acd42(52)*acd42(51)
      acd42(85)=acd42(54)*acd42(53)
      acd42(86)=acd42(56)*acd42(55)
      acd42(87)=acd42(58)*acd42(57)
      acd42(59)=acd42(87)+acd42(86)+acd42(85)+acd42(84)+acd42(83)+acd42(82)+acd&
      &42(81)+acd42(80)+acd42(79)+acd42(78)+acd42(77)+acd42(76)+acd42(75)+acd42&
      &(74)+acd42(73)+acd42(72)+acd42(71)+acd42(70)+acd42(69)+acd42(68)+acd42(6&
      &7)+acd42(66)+acd42(65)+acd42(64)+acd42(63)+acd42(62)+acd42(61)+acd42(59)&
      &+acd42(60)
      brack(ninjaidxt3mu0)=0.0_ki
      brack(ninjaidxt2mu0)=acd42(59)
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d42h8_ninja_t3")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd42h8
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a, b, c
      complex(ki), intent(in) :: param
      complex(ki), dimension(4) :: vecA, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      vecA(1:4) = + a(0:3)
      vecB(1:4) = + b(0:3)
      vecC(1:4) = + c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_31,vecA,vecB,vecC,param,coeffs)
   end subroutine numerator_t3
!---#] subroutine numerator_t3:
end module     p2_gg_httbar_d42h8l131
