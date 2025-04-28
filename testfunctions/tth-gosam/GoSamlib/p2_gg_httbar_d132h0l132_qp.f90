module     p2_gg_httbar_d132h0l132_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d132h0l132_qp.f90
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
      use p2_gg_httbar_abbrevd132h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(24) :: acd132
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd132(1)=dotproduct(ninjaE3,spvak2e2)
      acd132(2)=dotproduct(ninjaE3,spvae2k2)
      acd132(3)=abb132(13)
      acd132(4)=dotproduct(ninjaE3,spval5e2)
      acd132(5)=abb132(16)
      acd132(6)=dotproduct(ninjaE3,spval4e2)
      acd132(7)=abb132(17)
      acd132(8)=dotproduct(ninjaE3,spvak1e2)
      acd132(9)=abb132(19)
      acd132(10)=dotproduct(ninjaE3,spvae1e2)
      acd132(11)=abb132(27)
      acd132(12)=dotproduct(ninjaE3,spvae2l4)
      acd132(13)=abb132(18)
      acd132(14)=dotproduct(ninjaE3,spvae2k1)
      acd132(15)=abb132(22)
      acd132(16)=dotproduct(ninjaE3,spvae2l5)
      acd132(17)=abb132(24)
      acd132(18)=dotproduct(ninjaE3,spvae2e1)
      acd132(19)=abb132(88)
      acd132(20)=acd132(5)*acd132(2)
      acd132(21)=-acd132(13)*acd132(12)
      acd132(22)=acd132(15)*acd132(14)
      acd132(23)=acd132(17)*acd132(16)
      acd132(24)=-acd132(19)*acd132(18)
      acd132(20)=acd132(24)+acd132(23)+acd132(22)+acd132(21)+acd132(20)
      acd132(20)=acd132(4)*acd132(20)
      acd132(21)=acd132(3)*acd132(1)
      acd132(22)=acd132(7)*acd132(6)
      acd132(23)=acd132(9)*acd132(8)
      acd132(24)=acd132(11)*acd132(10)
      acd132(21)=acd132(24)+acd132(23)+acd132(22)+acd132(21)
      acd132(21)=acd132(2)*acd132(21)
      acd132(20)=acd132(20)+acd132(21)
      brack(ninjaidxt1x0mu0)=acd132(20)
   end subroutine brack_21
!---#] subroutine brack_21:
!---#[ subroutine brack_22:
   pure subroutine brack_22(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd132h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(62) :: acd132
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd132(1)=dotproduct(ninjaA1,spval5e2)
      acd132(2)=dotproduct(ninjaE3,spvae2k2)
      acd132(3)=abb132(16)
      acd132(4)=dotproduct(ninjaE3,spvae2k1)
      acd132(5)=abb132(22)
      acd132(6)=dotproduct(ninjaE3,spvae2l4)
      acd132(7)=abb132(18)
      acd132(8)=dotproduct(ninjaE3,spvae2l5)
      acd132(9)=abb132(24)
      acd132(10)=dotproduct(ninjaE3,spvae2e1)
      acd132(11)=abb132(88)
      acd132(12)=dotproduct(ninjaA1,spvak2e2)
      acd132(13)=abb132(13)
      acd132(14)=dotproduct(ninjaA1,spvae2k2)
      acd132(15)=dotproduct(ninjaE3,spval5e2)
      acd132(16)=dotproduct(ninjaE3,spvak2e2)
      acd132(17)=dotproduct(ninjaE3,spval4e2)
      acd132(18)=abb132(17)
      acd132(19)=dotproduct(ninjaE3,spvak1e2)
      acd132(20)=abb132(19)
      acd132(21)=dotproduct(ninjaE3,spvae1e2)
      acd132(22)=abb132(27)
      acd132(23)=dotproduct(ninjaA1,spvae2k1)
      acd132(24)=dotproduct(ninjaA1,spval4e2)
      acd132(25)=dotproduct(ninjaA1,spvae2l4)
      acd132(26)=dotproduct(ninjaA1,spvak1e2)
      acd132(27)=dotproduct(ninjaA1,spvae2l5)
      acd132(28)=dotproduct(ninjaA1,spvae1e2)
      acd132(29)=dotproduct(ninjaA1,spvae2e1)
      acd132(30)=dotproduct(ninjaA0,spval5e2)
      acd132(31)=dotproduct(ninjaA0,spvak2e2)
      acd132(32)=dotproduct(ninjaA0,spvae2k2)
      acd132(33)=dotproduct(ninjaA0,spvae2k1)
      acd132(34)=dotproduct(ninjaA0,spval4e2)
      acd132(35)=dotproduct(ninjaA0,spvae2l4)
      acd132(36)=dotproduct(ninjaA0,spvak1e2)
      acd132(37)=dotproduct(ninjaA0,spvae2l5)
      acd132(38)=dotproduct(ninjaA0,spvae1e2)
      acd132(39)=dotproduct(ninjaA0,spvae2e1)
      acd132(40)=abb132(12)
      acd132(41)=abb132(21)
      acd132(42)=abb132(14)
      acd132(43)=abb132(15)
      acd132(44)=abb132(39)
      acd132(45)=abb132(28)
      acd132(46)=abb132(23)
      acd132(47)=abb132(36)
      acd132(48)=abb132(43)
      acd132(49)=abb132(37)
      acd132(50)=-acd132(11)*acd132(29)
      acd132(51)=acd132(9)*acd132(27)
      acd132(52)=-acd132(7)*acd132(25)
      acd132(53)=acd132(5)*acd132(23)
      acd132(54)=acd132(14)*acd132(3)
      acd132(50)=acd132(54)+acd132(53)+acd132(52)+acd132(50)+acd132(51)
      acd132(50)=acd132(15)*acd132(50)
      acd132(51)=acd132(22)*acd132(28)
      acd132(52)=acd132(20)*acd132(26)
      acd132(53)=acd132(18)*acd132(24)
      acd132(54)=acd132(13)*acd132(12)
      acd132(55)=acd132(1)*acd132(3)
      acd132(51)=acd132(55)+acd132(54)+acd132(53)+acd132(51)+acd132(52)
      acd132(51)=acd132(2)*acd132(51)
      acd132(52)=acd132(22)*acd132(21)
      acd132(53)=acd132(20)*acd132(19)
      acd132(54)=acd132(18)*acd132(17)
      acd132(55)=acd132(13)*acd132(16)
      acd132(52)=acd132(52)+acd132(53)+acd132(54)+acd132(55)
      acd132(53)=acd132(14)*acd132(52)
      acd132(54)=acd132(11)*acd132(10)
      acd132(55)=acd132(9)*acd132(8)
      acd132(56)=acd132(7)*acd132(6)
      acd132(57)=acd132(5)*acd132(4)
      acd132(54)=acd132(54)-acd132(55)+acd132(56)-acd132(57)
      acd132(55)=-acd132(1)*acd132(54)
      acd132(50)=acd132(51)+acd132(50)+acd132(53)+acd132(55)
      acd132(51)=-acd132(11)*acd132(39)
      acd132(53)=acd132(9)*acd132(37)
      acd132(55)=-acd132(7)*acd132(35)
      acd132(56)=acd132(5)*acd132(33)
      acd132(57)=acd132(32)*acd132(3)
      acd132(51)=acd132(57)+acd132(56)+acd132(55)+acd132(53)+acd132(40)+acd132(&
      &51)
      acd132(51)=acd132(15)*acd132(51)
      acd132(53)=acd132(22)*acd132(38)
      acd132(55)=acd132(20)*acd132(36)
      acd132(56)=acd132(18)*acd132(34)
      acd132(57)=acd132(13)*acd132(31)
      acd132(58)=acd132(30)*acd132(3)
      acd132(53)=acd132(58)+acd132(57)+acd132(56)+acd132(55)+acd132(42)+acd132(&
      &53)
      acd132(53)=acd132(2)*acd132(53)
      acd132(52)=acd132(32)*acd132(52)
      acd132(54)=-acd132(30)*acd132(54)
      acd132(55)=acd132(21)*acd132(48)
      acd132(56)=acd132(19)*acd132(46)
      acd132(57)=acd132(17)*acd132(44)
      acd132(58)=acd132(16)*acd132(41)
      acd132(59)=acd132(10)*acd132(49)
      acd132(60)=acd132(8)*acd132(47)
      acd132(61)=acd132(6)*acd132(45)
      acd132(62)=acd132(4)*acd132(43)
      acd132(51)=acd132(53)+acd132(51)+acd132(54)+acd132(52)+acd132(62)+acd132(&
      &61)+acd132(60)+acd132(59)+acd132(58)+acd132(57)+acd132(55)+acd132(56)
      brack(ninjaidxt0x0mu0)=acd132(51)
      brack(ninjaidxt0x1mu0)=acd132(50)
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d132h0_qp_ninja_t2")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd132h0_qp
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a0, a1, b, c
      complex(ki), dimension(0:2), intent(in) :: param
      complex(ki), dimension(4) :: vecA0, vecA1, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = -k5
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
end module     p2_gg_httbar_d132h0l132_qp
