module     p0_gg_gh_d9h1l131_qp
   ! file: /mt/home/sjones/repos/POWHEG-BOX/ggh-gosam/GoSam_POWHEG/Virtual/p0_g &
   ! &g_gh/helicity1d9h1l131_qp.f90
   ! generator: buildfortran_tn3.py
   use p0_gg_gh_config, only: ki => ki_qp
   use p0_gg_gh_util_qp, only: cond_t, d => metric_tensor
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
      use p0_gg_gh_model_qp
      use p0_gg_gh_kinematics_qp
      use p0_gg_gh_color_qp
      use p0_gg_gh_abbrevd9h1_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(4) :: acd9
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd9(1)=dotproduct(ninjaE3,spvak2k1)
      acd9(2)=dotproduct(ninjaE3,spvak2k3)
      acd9(3)=abb9(6)
      acd9(4)=acd9(3)*acd9(1)*acd9(2)**2
      brack(ninjaidxt3mu0)=0.0_ki
      brack(ninjaidxt2mu0)=acd9(4)
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine brack_32:
   pure subroutine brack_32(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p0_gg_gh_model_qp
      use p0_gg_gh_kinematics_qp
      use p0_gg_gh_color_qp
      use p0_gg_gh_abbrevd9h1_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(21) :: acd9
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd9(1)=dotproduct(ninjaE3,spvak2k1)
      acd9(2)=dotproduct(ninjaE3,spvak2k3)
      acd9(3)=dotproduct(ninjaE4,spvak2k3)
      acd9(4)=abb9(6)
      acd9(5)=dotproduct(ninjaE4,spvak2k1)
      acd9(6)=dotproduct(ninjaA,spvak2k1)
      acd9(7)=dotproduct(ninjaA,spvak2k3)
      acd9(8)=abb9(8)
      acd9(9)=abb9(7)
      acd9(10)=dotproduct(ninjaE3,spvak2l4)
      acd9(11)=abb9(10)
      acd9(12)=dotproduct(ninjaA,spvak2l4)
      acd9(13)=abb9(11)
      acd9(14)=acd9(1)*acd9(4)
      acd9(15)=2.0_ki*acd9(2)
      acd9(16)=acd9(14)*acd9(15)
      acd9(17)=acd9(16)*acd9(3)
      acd9(18)=acd9(2)**2
      acd9(19)=acd9(4)*acd9(18)*acd9(5)
      acd9(17)=acd9(17)+acd9(19)
      acd9(19)=acd9(10)*acd9(11)
      acd9(20)=acd9(8)*acd9(1)
      acd9(20)=acd9(19)+acd9(20)
      acd9(20)=acd9(2)*acd9(20)
      acd9(21)=acd9(6)*acd9(4)
      acd9(21)=acd9(21)+acd9(9)
      acd9(18)=acd9(18)*acd9(21)
      acd9(16)=acd9(7)*acd9(16)
      acd9(16)=acd9(16)+acd9(18)+acd9(20)
      acd9(15)=acd9(15)*acd9(21)
      acd9(15)=acd9(19)+acd9(15)
      acd9(15)=acd9(7)*acd9(15)
      acd9(18)=acd9(7)*acd9(1)
      acd9(19)=acd9(6)*acd9(2)
      acd9(18)=acd9(18)+acd9(19)
      acd9(18)=acd9(8)*acd9(18)
      acd9(19)=ninjaP*acd9(17)
      acd9(20)=acd9(12)*acd9(11)
      acd9(20)=acd9(13)+acd9(20)
      acd9(20)=acd9(2)*acd9(20)
      acd9(14)=acd9(7)**2*acd9(14)
      acd9(14)=acd9(19)+acd9(18)+acd9(14)+acd9(15)+acd9(20)
      brack(ninjaidxt1mu0)=acd9(16)
      brack(ninjaidxt1mu2)=0.0_ki
      brack(ninjaidxt0mu0)=acd9(14)
      brack(ninjaidxt0mu2)=acd9(17)
   end subroutine brack_32
!---#] subroutine brack_32:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p0_gg_gh_d9h1_qp_ninja_t3")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p0_gg_gh_globalsl1_qp, only: epspow
      use p0_gg_gh_kinematics_qp
      use p0_gg_gh_abbrevd9h1_qp
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a, b, c
      complex(ki), intent(in) :: param
      complex(ki), dimension(4) :: vecA, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = k2
      vecA(1:4) = + a(0:3) - qshift(1:4)
      vecB(1:4) = + b(0:3)
      vecC(1:4) = + c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_31,vecA,vecB,vecC,param,coeffs)
      if (deg.le.(1+(0))) return
      call cond_t(epspow.eq.t1,brack_32,vecA,vecB,vecC,param,coeffs)
   end subroutine numerator_t3
!---#] subroutine numerator_t3:
end module     p0_gg_gh_d9h1l131_qp
