module     p0_gg_gh_d3h3l1_qp
   ! file: /mt/home/sjones/repos/POWHEG-BOX/ggh-gosam/GoSam_POWHEG/Virtual/p0_g &
   ! &g_gh/helicity3d3h3l1_qp.f90
   ! generator: buildfortran.py
   use p0_gg_gh_config, only: ki => ki_qp
   use p0_gg_gh_util_qp, only: cond
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   public :: numerator_ninja
contains
!---#[ function brack_1:
   pure function brack_1(Q,mu2) result(brack)
      use p0_gg_gh_model_qp
      use p0_gg_gh_kinematics_qp
      use p0_gg_gh_color_qp
      use p0_gg_gh_abbrevd3h3_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc3(16)
      complex(ki) :: Qspvak2k3
      complex(ki) :: Qspvak2k1
      complex(ki) :: Qspvak3k2
      complex(ki) :: Qspval4k2
      complex(ki) :: Qspvak3l4
      complex(ki) :: Qspvak3k1
      complex(ki) :: Qspk3
      complex(ki) :: Qspk2
      complex(ki) :: QspQ
      Qspvak2k3 = dotproduct(Q,spvak2k3)
      Qspvak2k1 = dotproduct(Q,spvak2k1)
      Qspvak3k2 = dotproduct(Q,spvak3k2)
      Qspval4k2 = dotproduct(Q,spval4k2)
      Qspvak3l4 = dotproduct(Q,spvak3l4)
      Qspvak3k1 = dotproduct(Q,spvak3k1)
      Qspk3 = dotproduct(Q,k3)
      Qspk2 = dotproduct(Q,k2)
      QspQ = dotproduct(Q,Q)
      acc3(1)=abb3(6)
      acc3(2)=abb3(7)
      acc3(3)=abb3(8)
      acc3(4)=abb3(9)
      acc3(5)=abb3(10)
      acc3(6)=abb3(11)
      acc3(7)=abb3(13)
      acc3(8)=abb3(15)
      acc3(9)=abb3(16)
      acc3(10)=Qspvak2k3*acc3(4)
      acc3(11)=Qspvak2k1*acc3(1)
      acc3(10)=acc3(11)+acc3(10)
      acc3(10)=Qspvak3k2*acc3(10)
      acc3(11)=Qspval4k2*acc3(2)
      acc3(12)=Qspvak3l4*acc3(3)
      acc3(13)=Qspvak3k1*acc3(5)
      acc3(14)=Qspk3*acc3(8)
      acc3(15)=Qspk2*acc3(9)
      acc3(16)=QspQ*acc3(6)
      brack=acc3(7)+acc3(10)+acc3(11)+acc3(12)+acc3(13)+acc3(14)+acc3(15)+acc3(&
      &16)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p0_gg_gh_d3h3l1_qp_ninja")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki_nin
      use p0_gg_gh_globalsl1_qp, only: epspow
      use p0_gg_gh_kinematics_qp
      use p0_gg_gh_abbrevd3h3_qp
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d3
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      Q(1:4)  =cmplx(real(+Q_ext(0:3),  ki_nin), aimag(+Q_ext(0:3)), ki)
      d3 = 0.0_ki
      d3 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d3, ki), aimag(d3), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p0_gg_gh_d3h3l1_qp
