module     p2_gg_httbar_d38h0l1_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d38h0l1_qp.f90
   ! generator: buildfortran.py
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_util_qp, only: cond
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   public :: numerator_ninja
contains
!---#[ function brack_1:
   pure function brack_1(Q,mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd38h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc38(47)
      complex(ki) :: Qspvae2e1
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspvae2l5
      complex(ki) :: Qspval5e2
      complex(ki) :: Qspvae1l5
      complex(ki) :: Qspval5e1
      complex(ki) :: Qspvae2l4
      complex(ki) :: Qspval4e2
      complex(ki) :: Qspvae2k2
      complex(ki) :: Qspvak2e2
      complex(ki) :: Qspvak2e1
      complex(ki) :: Qspvae2k1
      complex(ki) :: Qspvak1e2
      complex(ki) :: Qspval5l4
      complex(ki) :: Qspval5k2
      complex(ki) :: Qspval5k1
      complex(ki) :: Qspval4l5
      complex(ki) :: Qspvak2l5
      complex(ki) :: Qspvak2l4
      complex(ki) :: Qspvak2k1
      complex(ki) :: Qspvak1l5
      complex(ki) :: Qspl5
      complex(ki) :: Qspk2
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspvae2l5 = dotproduct(Q,spvae2l5)
      Qspval5e2 = dotproduct(Q,spval5e2)
      Qspvae1l5 = dotproduct(Q,spvae1l5)
      Qspval5e1 = dotproduct(Q,spval5e1)
      Qspvae2l4 = dotproduct(Q,spvae2l4)
      Qspval4e2 = dotproduct(Q,spval4e2)
      Qspvae2k2 = dotproduct(Q,spvae2k2)
      Qspvak2e2 = dotproduct(Q,spvak2e2)
      Qspvak2e1 = dotproduct(Q,spvak2e1)
      Qspvae2k1 = dotproduct(Q,spvae2k1)
      Qspvak1e2 = dotproduct(Q,spvak1e2)
      Qspval5l4 = dotproduct(Q,spval5l4)
      Qspval5k2 = dotproduct(Q,spval5k2)
      Qspval5k1 = dotproduct(Q,spval5k1)
      Qspval4l5 = dotproduct(Q,spval4l5)
      Qspvak2l5 = dotproduct(Q,spvak2l5)
      Qspvak2l4 = dotproduct(Q,spvak2l4)
      Qspvak2k1 = dotproduct(Q,spvak2k1)
      Qspvak1l5 = dotproduct(Q,spvak1l5)
      Qspl5 = dotproduct(Q,l5)
      Qspk2 = dotproduct(Q,k2)
      acc38(1)=abb38(14)
      acc38(2)=abb38(15)
      acc38(3)=abb38(16)
      acc38(4)=abb38(17)
      acc38(5)=abb38(18)
      acc38(6)=abb38(19)
      acc38(7)=abb38(20)
      acc38(8)=abb38(21)
      acc38(9)=abb38(22)
      acc38(10)=abb38(23)
      acc38(11)=abb38(24)
      acc38(12)=abb38(26)
      acc38(13)=abb38(27)
      acc38(14)=abb38(28)
      acc38(15)=abb38(30)
      acc38(16)=abb38(31)
      acc38(17)=abb38(32)
      acc38(18)=abb38(33)
      acc38(19)=abb38(34)
      acc38(20)=abb38(37)
      acc38(21)=abb38(38)
      acc38(22)=abb38(41)
      acc38(23)=abb38(44)
      acc38(24)=abb38(48)
      acc38(25)=Qspvae2e1*acc38(16)
      acc38(26)=Qspvae1e2*acc38(13)
      acc38(27)=Qspvae2l5*acc38(4)
      acc38(28)=Qspval5e2*acc38(24)
      acc38(29)=Qspvae1l5*acc38(8)
      acc38(30)=Qspval5e1*acc38(18)
      acc38(31)=Qspvae2l4*acc38(19)
      acc38(32)=Qspval4e2*acc38(23)
      acc38(33)=Qspvae2k2*acc38(1)
      acc38(34)=Qspvak2e2*acc38(6)
      acc38(35)=Qspvak2e1*acc38(10)
      acc38(36)=Qspvae2k1*acc38(15)
      acc38(37)=Qspvak1e2*acc38(7)
      acc38(38)=Qspval5l4*acc38(17)
      acc38(39)=Qspval5k2*acc38(20)
      acc38(40)=Qspval5k1*acc38(21)
      acc38(41)=Qspval4l5*acc38(22)
      acc38(42)=Qspvak2l5*acc38(2)
      acc38(43)=Qspvak2l4*acc38(9)
      acc38(44)=Qspvak2k1*acc38(11)
      acc38(45)=Qspvak1l5*acc38(5)
      acc38(46)=Qspl5*acc38(14)
      acc38(47)=Qspk2*acc38(3)
      brack=acc38(12)+acc38(25)+acc38(26)+acc38(27)+acc38(28)+acc38(29)+acc38(3&
      &0)+acc38(31)+acc38(32)+acc38(33)+acc38(34)+acc38(35)+acc38(36)+acc38(37)&
      &+acc38(38)+acc38(39)+acc38(40)+acc38(41)+acc38(42)+acc38(43)+acc38(44)+a&
      &cc38(45)+acc38(46)+acc38(47)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d38h0l1_qp_ninja")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd38h0_qp
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d38
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      Q(1:4)  =cmplx(real(+Q_ext(0:3),  ki_nin), aimag(+Q_ext(0:3)), ki)
      d38 = 0.0_ki
      d38 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d38, ki), aimag(d38), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d38h0l1_qp
