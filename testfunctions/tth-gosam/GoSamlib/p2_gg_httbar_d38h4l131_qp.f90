module     p2_gg_httbar_d38h4l131_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity4d38h4l131_qp.f90
   ! generator: buildfortran_tn3.py
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_util_qp, only: cond_t, d => metric_tensor
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
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd38h4_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(69) :: acd38
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd38(1)=dotproduct(k2,ninjaE3)
      acd38(2)=abb38(15)
      acd38(3)=dotproduct(l5,ninjaE3)
      acd38(4)=abb38(21)
      acd38(5)=dotproduct(ninjaE3,spval5k2)
      acd38(6)=abb38(14)
      acd38(7)=dotproduct(ninjaE3,spvak2l5)
      acd38(8)=abb38(16)
      acd38(9)=dotproduct(ninjaE3,spvae2k2)
      acd38(10)=abb38(17)
      acd38(11)=dotproduct(ninjaE3,spvak2e1)
      acd38(12)=abb38(18)
      acd38(13)=dotproduct(ninjaE3,spvae2k1)
      acd38(14)=abb38(20)
      acd38(15)=dotproduct(ninjaE3,spvae2e1)
      acd38(16)=abb38(22)
      acd38(17)=dotproduct(ninjaE3,spvae1e2)
      acd38(18)=abb38(23)
      acd38(19)=dotproduct(ninjaE3,spvak1e2)
      acd38(20)=abb38(24)
      acd38(21)=dotproduct(ninjaE3,spvak2e2)
      acd38(22)=abb38(25)
      acd38(23)=dotproduct(ninjaE3,spval5k1)
      acd38(24)=abb38(26)
      acd38(25)=dotproduct(ninjaE3,spval5l4)
      acd38(26)=abb38(27)
      acd38(27)=dotproduct(ninjaE3,spvak2k1)
      acd38(28)=abb38(28)
      acd38(29)=dotproduct(ninjaE3,spvak2l4)
      acd38(30)=abb38(29)
      acd38(31)=dotproduct(ninjaE3,spvak1l5)
      acd38(32)=abb38(30)
      acd38(33)=dotproduct(ninjaE3,spval4l5)
      acd38(34)=abb38(32)
      acd38(35)=dotproduct(ninjaE3,spvae2l5)
      acd38(36)=abb38(34)
      acd38(37)=dotproduct(ninjaE3,spvae1l5)
      acd38(38)=abb38(44)
      acd38(39)=dotproduct(ninjaE3,spval5e1)
      acd38(40)=abb38(50)
      acd38(41)=dotproduct(ninjaE3,spval5e2)
      acd38(42)=abb38(51)
      acd38(43)=dotproduct(ninjaE3,spvae2l4)
      acd38(44)=abb38(53)
      acd38(45)=dotproduct(ninjaE3,spval4e2)
      acd38(46)=abb38(58)
      acd38(47)=acd38(2)*acd38(1)
      acd38(48)=acd38(4)*acd38(3)
      acd38(49)=acd38(6)*acd38(5)
      acd38(50)=acd38(8)*acd38(7)
      acd38(51)=acd38(10)*acd38(9)
      acd38(52)=acd38(12)*acd38(11)
      acd38(53)=acd38(14)*acd38(13)
      acd38(54)=acd38(16)*acd38(15)
      acd38(55)=acd38(18)*acd38(17)
      acd38(56)=acd38(20)*acd38(19)
      acd38(57)=acd38(22)*acd38(21)
      acd38(58)=acd38(24)*acd38(23)
      acd38(59)=acd38(26)*acd38(25)
      acd38(60)=acd38(28)*acd38(27)
      acd38(61)=acd38(30)*acd38(29)
      acd38(62)=acd38(32)*acd38(31)
      acd38(63)=acd38(34)*acd38(33)
      acd38(64)=acd38(36)*acd38(35)
      acd38(65)=acd38(38)*acd38(37)
      acd38(66)=acd38(40)*acd38(39)
      acd38(67)=acd38(42)*acd38(41)
      acd38(68)=acd38(44)*acd38(43)
      acd38(69)=acd38(46)*acd38(45)
      acd38(47)=acd38(69)+acd38(68)+acd38(67)+acd38(66)+acd38(65)+acd38(64)+acd&
      &38(63)+acd38(62)+acd38(61)+acd38(60)+acd38(59)+acd38(58)+acd38(57)+acd38&
      &(56)+acd38(55)+acd38(54)+acd38(53)+acd38(52)+acd38(51)+acd38(50)+acd38(4&
      &9)+acd38(47)+acd38(48)
      brack(ninjaidxt3mu0)=0.0_ki
      brack(ninjaidxt2mu0)=acd38(47)
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d38h4_qp_ninja_t3")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd38h4_qp
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
end module     p2_gg_httbar_d38h4l131_qp
